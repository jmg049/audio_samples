//! Opus operating mode, audio bandwidth, and codec configuration types.
//!
//! ## Modes
//!
//! Opus defines three operating modes (RFC 6716 §2):
//!
//! | Mode   | Description |
//! |--------|-------------|
//! | `SILK` | LP codec for narrowband/wideband speech (Skype SILK). |
//! | `CELT` | MDCT codec for fullband music and generic audio (CELT/Xiph). |
//! | `Hybrid` | SILK for 0–8 kHz, CELT extension for 8–20 kHz. |
//!
//! The mode may be forced via [`OpusConfig::mode`] or set to `None` to trigger
//! automatic per-frame detection via [`detect_mode`].
//!
//! ## Sketch note
//!
//! Hybrid mode is represented in the type system but falls back to CELT in the
//! current [`crate::codecs::opus::OpusCodec`] implementation. Full hybrid
//! encoding (split-band LP + MDCT) is a planned extension.

// ── OpusMode ──────────────────────────────────────────────────────────────────

/// Operating mode of the Opus codec for a given audio frame.
///
/// See the [module documentation](self) for a comparison of modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpusMode {
    /// SILK speech codec: linear prediction coding.
    ///
    /// Best suited for narrowband and wideband speech signals. Uses an
    /// order-16 LPC predictor and 16-bit residual quantisation.
    Silk,

    /// CELT wideband audio codec: MDCT with psychoacoustic bit allocation.
    ///
    /// Best suited for fullband music, mixed content, and signals that resist
    /// LP prediction (e.g. white noise). Reuses the existing
    /// [`crate::codecs::perceptual`] pipeline.
    Celt,

    /// Hybrid mode: SILK for the baseband (0–8 kHz), CELT extension (8–20 kHz).
    ///
    /// Used in super-wideband and fullband speech encoding.
    ///
    /// > **Sketch note**: hybrid encoding is not yet fully implemented.
    /// > Frames detected as `Hybrid` are currently encoded with CELT.
    Hybrid,
}

// ── OpusBandwidth ─────────────────────────────────────────────────────────────

/// Audio bandwidth of an Opus stream.
///
/// Limits the highest reproduced frequency and constrains which modes are
/// applicable (RFC 6716 §2.1.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpusBandwidth {
    /// Narrowband: 0–4 kHz. Applicable mode: SILK only.
    NarrowBand,
    /// Mediumband: 0–6 kHz. Applicable mode: SILK only.
    MediumBand,
    /// Wideband: 0–8 kHz. Applicable modes: SILK or CELT.
    WideBand,
    /// Super-wideband: 0–12 kHz. Applicable modes: Hybrid or CELT.
    SuperWideBand,
    /// Fullband: 0–20 kHz. Applicable modes: Hybrid or CELT.
    FullBand,
}

impl OpusBandwidth {
    /// Returns the upper frequency limit in Hz for this bandwidth.
    #[inline]
    #[must_use]
    pub const fn max_frequency_hz(self) -> f32 {
        match self {
            Self::NarrowBand    => 4_000.0,
            Self::MediumBand    => 6_000.0,
            Self::WideBand      => 8_000.0,
            Self::SuperWideBand => 12_000.0,
            Self::FullBand      => 20_000.0,
        }
    }

    /// Returns `true` if SILK is a valid mode for this bandwidth.
    ///
    /// SILK is only applicable up to and including [`OpusBandwidth::WideBand`].
    #[inline]
    #[must_use]
    pub const fn supports_silk(self) -> bool {
        matches!(self, Self::NarrowBand | Self::MediumBand | Self::WideBand)
    }

    /// Returns `true` if CELT is a valid mode for this bandwidth.
    ///
    /// CELT requires at least [`OpusBandwidth::WideBand`] to be useful.
    #[inline]
    #[must_use]
    pub const fn supports_celt(self) -> bool {
        !matches!(self, Self::NarrowBand | Self::MediumBand)
    }
}

// ── OpusConfig ────────────────────────────────────────────────────────────────

/// Configuration parameters for the Opus sketch codec.
///
/// Passed to [`crate::codecs::opus::OpusCodec::new`] to control the operating
/// mode, bandwidth, bit budget, and frame length.
#[derive(Debug, Clone, PartialEq)]
pub struct OpusConfig {
    /// Forced operating mode for all frames.
    ///
    /// When `None`, the mode is detected automatically per frame via
    /// [`detect_mode`].
    pub mode: Option<OpusMode>,

    /// Audio bandwidth constraint.
    ///
    /// Determines the maximum reproduced frequency and restricts applicable
    /// modes. Use [`OpusBandwidth::FullBand`] for generic audio.
    pub bandwidth: OpusBandwidth,

    /// Total bit budget per CELT frame, or the overall coding budget for SILK
    /// residual quantisation.
    pub bit_budget: u32,

    /// Duration of one audio frame in milliseconds.
    ///
    /// Valid Opus values are 2.5, 5, 10, 20, 40, and 60 ms. Other positive
    /// values are accepted but may produce sub-optimal performance. Default: 20 ms.
    pub frame_size_ms: f32,

    /// Minimum bits guaranteed to every CELT band.
    ///
    /// Passed directly to [`crate::codecs::perceptual::quantization::allocate_bits`].
    /// Typical value: `1`.
    pub min_bits_per_band: u8,
}

impl OpusConfig {
    /// Creates an `OpusConfig` with automatic mode detection and fullband audio.
    ///
    /// Uses [`OpusBandwidth::FullBand`], a 20 ms frame, and `min_bits_per_band = 1`.
    ///
    /// # Arguments
    /// - `bit_budget` – Bit budget per frame.
    #[inline]
    #[must_use]
    pub fn new(bit_budget: u32) -> Self {
        Self {
            mode: None,
            bandwidth: OpusBandwidth::FullBand,
            bit_budget,
            frame_size_ms: 20.0,
            min_bits_per_band: 1,
        }
    }

    /// Creates an `OpusConfig` with a fixed operating mode.
    ///
    /// # Arguments
    /// - `mode` – Operating mode forced for every frame.
    /// - `bit_budget` – Bit budget per frame.
    #[inline]
    #[must_use]
    pub fn with_mode(mode: OpusMode, bit_budget: u32) -> Self {
        Self {
            mode: Some(mode),
            bandwidth: OpusBandwidth::FullBand,
            bit_budget,
            frame_size_ms: 20.0,
            min_bits_per_band: 1,
        }
    }
}

// ── Mode detection ────────────────────────────────────────────────────────────

/// Detects the appropriate Opus mode for an audio frame using the Spectral
/// Flatness Measure (SFM).
///
/// ## Algorithm
///
/// 1. Check bandwidth constraint: NarrowBand/MediumBand always → SILK; any
///    bandwidth that does not support SILK always → CELT.
/// 2. Compute the SFM of the frame's sub-band energy distribution:
///    `SFM = geometric_mean(band_energies) / arithmetic_mean(band_energies)`.
///    SFM ≈ 1 for flat/noise-like content; SFM ≈ 0 for tonal/speech content.
/// 3. Apply thresholds:
///    - SFM > 0.8 → [`OpusMode::Celt`]
///    - SFM < 0.4 → [`OpusMode::Silk`]
///    - Otherwise → [`OpusMode::Hybrid`]
///
/// # Arguments
/// - `samples` – One audio frame (typically 20 ms of PCM, f32).
/// - `_sample_rate` – Sample rate in Hz (reserved for future use).
/// - `bandwidth` – Bandwidth constraint from [`OpusConfig`].
#[must_use]
pub fn detect_mode(samples: &[f32], _sample_rate: u32, bandwidth: OpusBandwidth) -> OpusMode {
    // Bandwidth hard-constraints.
    if !bandwidth.supports_celt() {
        return OpusMode::Silk;
    }
    if !bandwidth.supports_silk() {
        return OpusMode::Celt;
    }

    let n = samples.len();
    if n < 4 {
        return OpusMode::Celt;
    }

    // Split frame into 8 sub-bands and compute SFM over band energies.
    const N_SUB_BANDS: usize = 8;
    let band_size = n / N_SUB_BANDS;
    if band_size == 0 {
        return OpusMode::Celt;
    }

    let band_energies: Vec<f64> = (0..N_SUB_BANDS)
        .map(|b| {
            let start = b * band_size;
            let end = ((b + 1) * band_size).min(n);
            samples[start..end]
                .iter()
                .map(|&x| (x as f64).powi(2))
                .sum::<f64>()
                + 1e-10 // guard against log(0)
        })
        .collect();

    let arith_mean = band_energies.iter().sum::<f64>() / N_SUB_BANDS as f64;
    let log_sum: f64 = band_energies.iter().map(|&e| e.ln()).sum::<f64>();
    let geom_mean = (log_sum / N_SUB_BANDS as f64).exp();

    let sfm = (geom_mean / arith_mean) as f32;

    if sfm > 0.8 {
        OpusMode::Celt
    } else if sfm < 0.4 {
        OpusMode::Silk
    } else {
        OpusMode::Hybrid
    }
}
