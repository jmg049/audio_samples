//! Psychoacoustic masking model.
//!
//! Implements the three core steps of a perceptual masking model:
//!
//! 1. **Absolute threshold of hearing (ATH)** — the minimum audible level at each
//!    frequency, below which masking is never needed.
//! 2. **Spreading function** — how a loud masker raises the masking threshold in
//!    neighbouring Bark bands (simultaneous masking). Tonal maskers (sinusoids)
//!    are more effective than noise maskers and use a higher gain value.
//! 3. **Masking-threshold computation** — combines ATH, band energies, masker
//!    classification, and the spreading function to produce a per-band threshold
//!    and signal-to-mask ratio.
//!
//! These primitives follow the structure of the MPEG-1 psychoacoustic model (ISO
//! 11172-3 Annex D), simplified for use as a general-purpose analysis layer.

use super::{BandLayout, BandMetric, BandMetrics, PsychoacousticConfig};
use non_empty_slice::NonEmptySlice;

// ── Masker type ────────────────────────────────────────────────────────────────

/// Classification of a spectral masker as tonal (sinusoidal) or noise-like.
///
/// The masking effectiveness differs between masker types: tonal maskers (pure
/// tones, harmonics) create a tighter but deeper masking threshold, while noise
/// maskers create a broader but shallower threshold. MPEG-1 uses 14.5 dB offset
/// for tonal maskers and 5.5 dB for noise maskers.
///
/// Determined per-band by [`classify_masker_types`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaskerType {
    /// Tonal masker (sinusoidal, narrow-band). Uses `config.masking_gain`.
    Tonal,
    /// Noise-like masker (broadband). Uses `config.noise_masking_gain`.
    Noise,
}

/// Classifies each band's masker type based on local spectral prominence.
///
/// A band is classified as [`MaskerType::Tonal`] if its energy is at least
/// `tonal_threshold_db` above the average energy of its two immediate neighbours.
/// Bands with no neighbours (the first and last) are compared against their single
/// available neighbour. All other bands are classified as [`MaskerType::Noise`].
///
/// # Arguments
/// - `band_energy_db` – Per-band energy in dB, in band order.
/// - `tonal_threshold_db` – Minimum prominence (in dB) for a band to be tonal.
///   Typical value: `7.0` dB.
///
/// # Returns
/// A `Vec<MaskerType>` of the same length as `band_energy_db`.
#[must_use]
pub fn classify_masker_types(band_energy_db: &[f32], tonal_threshold_db: f32) -> Vec<MaskerType> {
    let n = band_energy_db.len();
    (0..n)
        .map(|i| {
            let neighbours: f32 = match (i.checked_sub(1), (i + 1 < n).then_some(i + 1)) {
                (Some(l), Some(r)) => (band_energy_db[l] + band_energy_db[r]) / 2.0,
                (Some(l), None) => band_energy_db[l],
                (None, Some(r)) => band_energy_db[r],
                (None, None) => band_energy_db[i],
            };
            if band_energy_db[i] - neighbours >= tonal_threshold_db {
                MaskerType::Tonal
            } else {
                MaskerType::Noise
            }
        })
        .collect()
}

// ── Absolute threshold of hearing ────────────────────────────────────────────

/// Returns the absolute threshold of hearing at `freq_hz` in dB SPL.
///
/// Uses the ISO 226 / MPEG-1 polynomial approximation:
///
/// ```text
/// ATH(f) = 3.64·(f/1000)^−0.8 − 6.5·exp(−0.6·(f/1000 − 3.3)²) + 10⁻³·(f/1000)⁴
/// ```
///
/// # Arguments
/// - `freq_hz` – Frequency in hertz. Values ≤ 0 return 60 dB (conservative maximum).
///
/// # Returns
/// Threshold in dB SPL, clamped to [−20, 60].
#[inline]
#[must_use]
pub fn absolute_threshold_of_hearing(freq_hz: f32) -> f32 {
    if freq_hz <= 0.0 {
        return 60.0;
    }
    let f = freq_hz / 1000.0;
    let term1 = 3.64_f32 * f.powf(-0.8_f32);
    let term2 = -6.5_f32 * (-0.6_f32 * (f - 3.3_f32).powi(2)).exp();
    let term3 = 1e-3_f32 * f.powi(4);
    (term1 + term2 + term3).clamp(-20.0, 60.0)
}

// ── Spreading function ────────────────────────────────────────────────────────

/// Returns the total masking attenuation in dB between a masker at `masker_bark`
/// and a maskee at `maskee_bark`.
///
/// The masking threshold the masker creates at `maskee_bark` is:
/// `masker_level_db − spreading_attenuation(masker_bark, maskee_bark, masker_type, config)`.
/// This means:
///
/// - At zero distance: attenuation equals the masker-type gain. For tonal maskers
///   this is `config.masking_gain` (e.g. 14.5 dB); for noise maskers it is
///   `config.noise_masking_gain` (e.g. 5.5 dB). A larger gain means the threshold
///   sits further below the masker, so the masker is more effective.
/// - With increasing Bark distance: attenuation grows by `upward_spread` dB/Bark
///   upward and `downward_spread` dB/Bark downward, reducing the masker's reach.
///
/// A signal is audible when its SMR (energy − threshold) is positive.
///
/// # Arguments
/// - `masker_bark` – Bark position of the masking tone or band.
/// - `maskee_bark` – Bark position of the masked band.
/// - `masker_type` – Whether the masker is tonal or noise-like.
/// - `config` – Psychoacoustic configuration containing spread slopes and gains.
///
/// # Returns
/// Total attenuation in dB.
#[inline]
#[must_use]
pub fn spreading_attenuation(
    masker_bark: f32,
    maskee_bark: f32,
    masker_type: MaskerType,
    config: &PsychoacousticConfig,
) -> f32 {
    let base_gain = match masker_type {
        MaskerType::Tonal => config.masking_gain,
        MaskerType::Noise => config.noise_masking_gain,
    };
    let dz = maskee_bark - masker_bark;
    let distance_decay = if dz >= 0.0 {
        config.upward_spread * dz
    } else {
        config.downward_spread * (-dz)
    };
    base_gain + distance_decay
}

// ── Band-metric computation ───────────────────────────────────────────────────

/// Computes per-band psychoacoustic metrics from a linear-scale bin-energy array.
///
/// Steps:
/// 1. Aggregate bin energies within each band's `[start_bin, end_bin)` range,
///    normalised by band width.
/// 2. Convert aggregated energy to dB, applying `config.epsilon` as a noise floor.
/// 3. Compute ATH for each band's centre frequency.
/// 4. For each target band `j`, the masking threshold = max(ATH[j], max over all
///    bands `i` of `energy[i] − spreading_attenuation(i → j)`). Bands far from
///    the masker contribute a lower threshold; the dominant masker wins.
/// 5. Derive SMR, importance (perceptual weight × SMR), and allowed noise.
///
/// # Arguments
/// - `bin_energies` – Linear power per spectral bin (e.g. `|MDCT[k]|²`).
/// - `band_layout` – Band definitions mapping bins to perceptual positions.
/// - `config` – Psychoacoustic model parameters.
/// - `n_bins` – Total number of spectral bins; must match the length of
///   `bin_energies`.
///
/// # Returns
/// A [`BandMetrics`] with one [`BandMetric`] per band in `band_layout`.
///
/// # Panics
///
/// Panics if `n_bins == 0` or if `config.perceptual_weights` length does not
/// match `band_layout.len()` (use [`PsychoacousticConfig::is_compatible_with`]
/// to validate beforehand).
pub fn compute_band_metrics(
    bin_energies: &[f32],
    band_layout: &BandLayout,
    config: &PsychoacousticConfig,
    n_bins: usize,
) -> BandMetrics {
    use super::bands::hz_to_bark;

    let bands = band_layout.as_slice();
    let n_bands = bands.len().get();
    let weights = config.perceptual_weights.as_non_empty_slice();

    // Step 1 & 2: per-band energy in dB.
    let band_energy_db: Vec<f32> = bands
        .iter()
        .map(|band| {
            let start = band.start_bin.min(n_bins);
            let end = band.end_bin.min(n_bins);
            let width = (end - start).max(1);
            let sum: f32 = bin_energies[start..end].iter().sum();
            10.0_f32 * (sum / width as f32 + config.epsilon).log10()
        })
        .collect();

    // Step 3: ATH per band.
    let ath_db: Vec<f32> = bands
        .iter()
        .map(|band| absolute_threshold_of_hearing(band.centre_frequency))
        .collect();

    // Bark positions for spreading computation.
    let bark_positions: Vec<f32> = bands
        .iter()
        .map(|band| hz_to_bark(band.centre_frequency))
        .collect();

    // Classify each band as tonal or noise-like (7 dB prominence threshold).
    let masker_types = classify_masker_types(&band_energy_db, 7.0);

    // Step 4: masking threshold per band.
    // For each target band j, find the strongest masking contribution from
    // any band i: threshold_contribution(i→j) = energy[i] − attenuation(i,j).
    // The dominant contribution determines the masking threshold (floored by ATH).
    let masking_thresholds: Vec<f32> = (0..n_bands)
        .map(|j| {
            let dominated_by: f32 = (0..n_bands)
                .map(|i| {
                    let attenuation = spreading_attenuation(
                        bark_positions[i],
                        bark_positions[j],
                        masker_types[i],
                        config,
                    );
                    band_energy_db[i] - attenuation
                })
                .fold(f32::NEG_INFINITY, f32::max);
            dominated_by.max(ath_db[j])
        })
        .collect();

    // Step 5: per-band metrics.
    let metrics: Vec<BandMetric> = (0..n_bands)
        .map(|i| {
            let energy = band_energy_db[i];
            let masking_threshold = masking_thresholds[i];
            let smr = compute_smr(energy, masking_threshold);
            let weight = weights[i];
            let importance = weight * smr.max(0.0);
            let allowed_noise = masking_threshold - smr.max(0.0);
            BandMetric::new(
                bands[i].clone(),
                energy,
                masking_threshold,
                smr,
                importance,
                allowed_noise,
            )
        })
        .collect();

    // SAFETY: n_bands >= 1 (BandLayout invariant) so metrics is non-empty.
    let metrics_slice = unsafe { NonEmptySlice::new_unchecked(&metrics) };
    BandMetrics::new(metrics_slice)
}

// ── Temporal masking ──────────────────────────────────────────────────────────

/// Applies temporal (time-domain) masking to a sequence of per-frame band metrics.
///
/// Simultaneous masking (computed by [`compute_band_metrics`]) only considers
/// energy present in the same MDCT frame. Temporal masking extends this:
///
/// - **Post-masking (forward masking)**: a loud event at frame `t` elevates the
///   masking threshold in subsequent frames for roughly 100–200 ms. The threshold
///   decays at `post_masking_decay_db_per_ms` dB per millisecond.
/// - **Pre-masking (backward masking)**: a loud event at frame `t+1` slightly
///   raises the threshold at earlier frames for roughly 20 ms. This is weaker
///   and decays faster at `pre_masking_decay_db_per_ms` dB per millisecond.
///
/// Both effects propagate until the contribution falls below the already-computed
/// simultaneous threshold, so the simultaneous threshold is always the lower bound.
///
/// # Arguments
/// - `frame_metrics` – Per-frame [`BandMetrics`] in temporal order, as produced by
///   repeated [`compute_band_metrics`] calls. Returns an empty `Vec` if empty.
/// - `config` – Psychoacoustic config; used to re-derive `importance` and
///   `allowed_noise` with the updated thresholds.
/// - `hop_duration_ms` – Duration of each MDCT hop in milliseconds
///   (`hop_size / sample_rate × 1000`).
/// - `post_masking_decay_db_per_ms` – Forward masking decay rate.
///   Typical MPEG-1 value: `0.15` dB/ms.
/// - `pre_masking_decay_db_per_ms` – Backward masking decay rate.
///   Typical MPEG-1 value: `3.0` dB/ms.
///
/// # Returns
/// A `Vec<BandMetrics>` of the same length with masking thresholds, SMR,
/// importance, and allowed noise updated to reflect temporal spread.
///
/// # Panics
///
/// Panics if any `BandMetrics` in `frame_metrics` has a different band count
/// than the first frame, or if `config.perceptual_weights` has fewer entries
/// than the number of bands.
pub fn apply_temporal_masking(
    frame_metrics: &[BandMetrics],
    config: &PsychoacousticConfig,
    hop_duration_ms: f32,
    post_masking_decay_db_per_ms: f32,
    pre_masking_decay_db_per_ms: f32,
) -> Vec<BandMetrics> {
    if frame_metrics.is_empty() {
        return Vec::new();
    }

    let n_frames = frame_metrics.len();
    let n_bands = frame_metrics[0].metrics.len().get();

    // Work on a mutable 2D grid [frame][band] of masking thresholds.
    // Initialise from the simultaneous-masking thresholds.
    let mut thresholds: Vec<Vec<f32>> = frame_metrics
        .iter()
        .map(|fm| fm.metrics.iter().map(|m| m.masking_threshold).collect())
        .collect();

    // Post-masking: frame t propagates forward until decay removes the contribution.
    for t in 0..n_frames {
        for b in 0..n_bands {
            let masker = frame_metrics[t].metrics[b].masking_threshold;
            let mut k = 1usize;
            loop {
                let future = t + k;
                if future >= n_frames {
                    break;
                }
                let decay = post_masking_decay_db_per_ms * (k as f32 * hop_duration_ms);
                let contribution = masker - decay;
                if contribution <= thresholds[future][b] {
                    break;
                }
                thresholds[future][b] = contribution;
                k += 1;
            }
        }
    }

    // Pre-masking: frame t propagates backward (weaker, faster decay).
    for t in 1..n_frames {
        for b in 0..n_bands {
            let masker = frame_metrics[t].metrics[b].masking_threshold;
            let mut k = 1usize;
            loop {
                if k > t {
                    break;
                }
                let past = t - k;
                let decay = pre_masking_decay_db_per_ms * (k as f32 * hop_duration_ms);
                let contribution = masker - decay;
                if contribution <= thresholds[past][b] {
                    break;
                }
                thresholds[past][b] = contribution;
                k += 1;
            }
        }
    }

    // Rebuild BandMetrics with updated derived fields.
    frame_metrics
        .iter()
        .zip(thresholds.iter())
        .map(|(fm, thresh_row)| {
            let metrics: Vec<BandMetric> = fm
                .metrics
                .iter()
                .zip(thresh_row.iter())
                .enumerate()
                .map(|(b, (m, &new_threshold))| {
                    let energy = m.energy;
                    let smr = compute_smr(energy, new_threshold);
                    let weight = config.perceptual_weights[b];
                    let importance = weight * smr.max(0.0);
                    let allowed_noise = new_threshold - smr.max(0.0);
                    BandMetric::new(
                        m.band.clone(),
                        energy,
                        new_threshold,
                        smr,
                        importance,
                        allowed_noise,
                    )
                })
                .collect();
            // SAFETY: n_bands >= 1 (BandMetrics invariant, preserved from input).
            let metrics_ne = unsafe { non_empty_slice::NonEmptySlice::new_unchecked(&metrics) };
            BandMetrics::new(metrics_ne)
        })
        .collect()
}

// ── Transient detection ───────────────────────────────────────────────────────

/// Detects transient windows in a mono signal by comparing short-term energy.
///
/// A window is classified as a transient if its RMS power exceeds the previous
/// window's RMS power by more than `threshold_ratio`. This identifies sudden
/// energy onsets (attacks, percussive events) where short windows are needed
/// to prevent pre-echo.
///
/// The first window is never a transient (no prior reference).
///
/// # Arguments
/// - `samples` – Mono f32 signal samples.
/// - `window_size` – Analysis window width in samples.
/// - `hop_size` – Hop between consecutive windows (typically `window_size / 2`).
/// - `threshold_ratio` – Energy ratio that triggers a transient flag.
///   Values of `6.0`–`10.0` correspond to roughly 8–10 dB sudden onsets.
///
/// # Returns
/// A [`NonEmptyVec<bool>`] with one entry per window; `true` = transient detected.
///
/// # Panics
///
/// Panics if `window_size` or `hop_size` is zero (both are `NonZeroUsize`).
#[must_use]
pub fn detect_transient_windows(
    samples: &non_empty_slice::NonEmptySlice<f32>,
    window_size: std::num::NonZeroUsize,
    hop_size: std::num::NonZeroUsize,
    threshold_ratio: f32,
) -> non_empty_slice::NonEmptyVec<bool> {
    let ws = window_size.get();
    let hs = hop_size.get();
    let n = samples.len().get();

    let energies: Vec<f32> = (0..)
        .map(|i: usize| i * hs)
        .take_while(|&pos| pos < n)
        .map(|pos| {
            let end = (pos + ws).min(n);
            let w = &samples[pos..end];
            w.iter().map(|s| s * s).sum::<f32>() / w.len() as f32
        })
        .collect();

    let result: Vec<bool> = energies
        .iter()
        .enumerate()
        .map(|(i, &e)| i > 0 && energies[i - 1] > 1e-10 && e > energies[i - 1] * threshold_ratio)
        .collect();

    // SAFETY: at least one window since n >= 1 (NonEmptySlice).
    unsafe { non_empty_slice::NonEmptyVec::new_unchecked(result) }
}

/// Computes the signal-to-mask ratio for a single band.
///
/// SMR = band energy (dB) − masking threshold (dB). A positive value means the
/// band is audible above the masking threshold; a negative value means it is
/// fully masked.
///
/// # Arguments
/// - `band_energy_db` – Aggregated band energy in dB.
/// - `masking_threshold_db` – Masking threshold for the band in dB.
///
/// # Returns
/// SMR in dB.
#[inline]
#[must_use]
pub fn compute_smr(band_energy_db: f32, masking_threshold_db: f32) -> f32 {
    band_energy_db - masking_threshold_db
}
