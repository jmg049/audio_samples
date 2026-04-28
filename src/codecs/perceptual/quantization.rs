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
//! `importance` and `allowed_noise` scores, but does not act on them. This
//! module bridges analysis and coding: `importance` drives how many bits each
//! band receives, while `allowed_noise` determines the coarseness of the
//! quantizer so that quantization noise stays below the masking threshold.
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

// ── Per-band allocation ───────────────────────────────────────────────────────

/// Bit allocation and quantization step size for a single frequency band.
#[derive(Debug, Clone, PartialEq)]
pub struct BandAllocation {
    /// First spectral bin index (inclusive) from the source [`Band`](super::Band).
    pub start_bin: usize,
    /// One past the last spectral bin index (exclusive).
    pub end_bin: usize,
    /// Bits allocated to this band (0 = band is not coded).
    pub bits: u8,
    /// Uniform quantization step size in linear amplitude.
    ///
    /// Chosen so that RMS quantization noise stays below the psychoacoustic
    /// masking threshold: `step_size = 10^(allowed_noise_db / 20) × √12`.
    pub step_size: f32,
}

/// Per-band bit allocation derived from [`BandMetrics`].
#[derive(Debug, Clone, PartialEq)]
pub struct BitAllocationResult {
    /// One [`BandAllocation`] per frequency band, in the same order as the
    /// source [`BandMetrics`].
    pub allocations: NonEmptyVec<BandAllocation>,
}

// ── Step size helper ──────────────────────────────────────────────────────────

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

// ── Bit allocation ────────────────────────────────────────────────────────────

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

    let total_importance: f32 = band_metrics
        .metrics
        .iter()
        .map(|m| m.importance.max(0.0))
        .sum();

    let allocations: Vec<BandAllocation> = band_metrics
        .metrics
        .iter()
        .map(|m| {
            let extra_bits = if total_importance > 0.0 {
                let fraction = m.importance.max(0.0) / total_importance;
                (fraction * remaining as f32).round() as u8
            } else {
                0
            };
            let bits = min_bits_per_band.saturating_add(extra_bits);
            let step_size = step_size_from_allowed_noise(m.allowed_noise);
            BandAllocation {
                start_bin: m.band.start_bin,
                end_bin: m.band.end_bin,
                bits,
                step_size,
            }
        })
        .collect();

    // SAFETY: n_bands >= 1 (BandMetrics invariant).
    let allocations = unsafe { NonEmptyVec::new_unchecked(allocations) };
    BitAllocationResult { allocations }
}

// ── Per-band helpers ──────────────────────────────────────────────────────────

/// Uniformly quantizes a slice of `f32` MDCT coefficients to `i32` indices.
///
/// `q[i] = round(c[i] / step_size)`
///
/// # Arguments
/// - `coefficients` – Float coefficients for a single band.
/// - `step_size` – Quantization step size (from [`BandAllocation::step_size`]).
#[inline]
#[must_use]
pub fn quantize_band(coefficients: &[f32], step_size: f32) -> Vec<i32> {
    coefficients
        .iter()
        .map(|&c| (c / step_size).round() as i32)
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

// ── Full-matrix quantization ─────────────────────────────────────────────────

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
        for k in alloc.start_bin..alloc.end_bin.min(nc) {
            for f in 0..nf {
                let idx = k * nf + f;
                out[idx] = (coefficients[idx] / step).round() as i32;
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
