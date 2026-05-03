//! Perceptual band layout constructors and frequency-scale helpers.
//!
//! This module provides preset [`BandLayout`] constructors that partition the
//! audible spectrum into perceptually-motivated bands, along with the
//! conversion helpers that underpin them.
//!
//! Two scales are supported:
//!
//! - **Bark** – 24 critical bands derived from the Traunmüller (1990) formula.
//!   Closely models the ear's frequency resolution and is the foundation for
//!   most MPEG psychoacoustic models.
//! - **Mel** – linearly-spaced on the Mel scale (O'Shaughnessy formula).
//!   Commonly used for speech and music feature extraction.
//!
//! Each [`Band`] produced by these constructors stores:
//! - `start_bin` / `end_bin` — indices into the caller's frequency-bin array
//!   (MDCT or FFT, whichever was passed as `n_bins`).
//! - `centre_frequency` — centre frequency in Hz.
//! - `perceptual_position` — normalised position in [0, 1] on the respective scale.
//!
//! ## Example
//!
//! ```rust
//! # #[cfg(feature = "psychoacoustic")] {
//! use audio_samples::BandLayout;
//! use std::num::NonZeroUsize;
//!
//! // 24 Bark bands for a 44.1 kHz signal with 1024 MDCT bins.
//! let layout = BandLayout::bark(
//!     NonZeroUsize::new(24).unwrap(),
//!     44100.0,
//!     NonZeroUsize::new(1024).unwrap(),
//! );
//! assert_eq!(layout.len().get(), 24);
//! # }
//! ```

use std::num::NonZeroUsize;

use non_empty_slice::NonEmptySlice;

use super::{Band, BandLayout};

// ── Bark-scale helpers ────────────────────────────────────────────────────────

/// Converts a frequency in Hz to its Bark-scale value.
///
/// Uses the Traunmüller (1990) formula:
/// `bark = 26.81 * hz / (1960 + hz) − 0.53`
///
/// # Arguments
/// - `hz` – Frequency in hertz. Values ≤ 0 are clamped to 0 Hz before conversion.
///
/// # Returns
/// Bark value, clamped to a minimum of 0.0.
#[inline]
#[must_use]
pub fn hz_to_bark(hz: f32) -> f32 {
    if hz <= 0.0 {
        return 0.0;
    }
    (26.81_f32 * hz / (1960.0_f32 + hz) - 0.53_f32).max(0.0)
}

/// Converts a Bark-scale value back to a frequency in Hz.
///
/// Inverse of the Traunmüller (1990) formula.
///
/// # Arguments
/// - `bark` – Bark value. Values < 0 are clamped to 0 before conversion.
///
/// # Returns
/// Frequency in hertz, clamped to a minimum of 0.0 Hz.
#[inline]
#[must_use]
pub fn bark_to_hz(bark: f32) -> f32 {
    let bark = bark.max(0.0);
    (1960.0_f32 * (bark + 0.53_f32) / (26.28_f32 - bark)).max(0.0)
}

// ── Mel-scale helpers ─────────────────────────────────────────────────────────

/// Converts a frequency in Hz to its Mel-scale value.
///
/// Uses the O'Shaughnessy formula: `mel = 2595 × log10(1 + hz / 700)`.
///
/// # Arguments
/// - `hz` – Frequency in hertz. Values ≤ 0 are clamped to 0 Hz.
///
/// # Returns
/// Mel value, clamped to a minimum of 0.0.
#[inline]
#[must_use]
pub fn hz_to_mel(hz: f32) -> f32 {
    if hz <= 0.0 {
        return 0.0;
    }
    2595.0_f32 * (1.0_f32 + hz / 700.0_f32).log10()
}

/// Converts a Mel-scale value back to a frequency in Hz.
///
/// Inverse of the O'Shaughnessy formula: `hz = 700 × (10^(mel / 2595) − 1)`.
///
/// # Arguments
/// - `mel` – Mel value. Values < 0 are clamped to 0.
///
/// # Returns
/// Frequency in hertz, clamped to a minimum of 0.0 Hz.
#[inline]
#[must_use]
pub fn mel_to_hz(mel: f32) -> f32 {
    if mel <= 0.0 {
        return 0.0;
    }
    (700.0_f32 * (10.0_f32.powf(mel / 2595.0_f32) - 1.0_f32)).max(0.0)
}

// ── Shared bin-mapping helper ─────────────────────────────────────────────────

/// Maps a frequency in Hz to the nearest integer bin index.
///
/// Assumes a linear spacing from 0 Hz (bin 0) to Nyquist (bin `n_bins − 1`).
/// The result is clamped to `[0, n_bins − 1]`.
#[inline]
fn hz_to_bin(hz: f32, sample_rate_hz: f32, n_bins: usize) -> usize {
    let nyquist = sample_rate_hz / 2.0;
    let bin = (hz / nyquist * (n_bins - 1) as f32).round() as isize;
    bin.clamp(0, (n_bins - 1) as isize) as usize
}

// ── ERB-scale helpers ─────────────────────────────────────────────────────────

/// Converts a frequency in Hz to its ERB-bandwidth value (Glasberg & Moore, 1990).
///
/// `ERB(f) = 24.7 × (4.37 × f/1000 + 1)`
///
/// This gives the bandwidth of the equivalent rectangular auditory filter at `f`.
/// At 0 Hz the minimum value is 24.7. The function is linear in Hz, which makes
/// it a good approximation for spacing perceptual filters from low to high frequency.
///
/// # Arguments
/// - `hz` – Frequency in hertz. Negative values are clamped to 0.
///
/// # Returns
/// ERB-bandwidth value (always ≥ 24.7).
#[inline]
#[must_use]
pub fn hz_to_erb(hz: f32) -> f32 {
    24.7_f32 * (4.37_f32 * hz.max(0.0) / 1000.0_f32 + 1.0_f32)
}

/// Converts an ERB-bandwidth value back to a frequency in Hz.
///
/// Inverse of the Glasberg & Moore (1990) formula:
/// `f = 1000 × (erb/24.7 − 1) / 4.37`
///
/// # Arguments
/// - `erb` – ERB-bandwidth value. Values below 24.7 (the minimum, corresponding to 0 Hz)
///   are clamped to produce 0 Hz.
///
/// # Returns
/// Frequency in hertz, clamped to a minimum of 0.0.
#[inline]
#[must_use]
pub fn erb_to_hz(erb: f32) -> f32 {
    (1000.0_f32 * (erb / 24.7_f32 - 1.0_f32) / 4.37_f32).max(0.0)
}

// ── Derived layout helpers ────────────────────────────────────────────────────

/// Scales a [`BandLayout`] from one spectral-bin count to another.
///
/// Each band's `start_bin` and `end_bin` are scaled proportionally from
/// `from_n_bins` to `to_n_bins`, preserving the frequency mapping while
/// adapting to a different MDCT window size. This is used when window
/// switching re-analyses a transient region with a shorter window that
/// produces fewer spectral bins.
///
/// # Arguments
/// - `layout` – Layout to scale.
/// - `from_n_bins` – Bin count the layout was built for.
/// - `to_n_bins` – Target bin count.
///
/// # Returns
/// A new [`BandLayout`] with the same number of bands and scaled bin ranges.
#[must_use]
pub fn scale_band_layout(layout: &BandLayout, from_n_bins: NonZeroUsize, to_n_bins: NonZeroUsize) -> BandLayout {
    let from = from_n_bins.get();
    let to = to_n_bins.get();

    let bands: Vec<super::Band> = layout.as_slice().iter().map(|band| {
        let start = band.start_bin * to / from;
        let end = (band.end_bin * to / from).max(start + 1).min(to);
        // SAFETY: end > start is enforced by the .max(start + 1) above.
        unsafe { super::Band::new(start, end, band.centre_frequency, band.perceptual_position) }
    }).collect();

    // SAFETY: layout is non-empty (BandLayout invariant), so bands is non-empty.
    let ne = unsafe { NonEmptySlice::new_unchecked(&bands) };
    BandLayout::new(ne)
}

// ── BandLayout preset constructors ───────────────────────────────────────────

impl BandLayout {
    /// Creates a Bark-scale band layout.
    ///
    /// Divides the spectrum from 0 Hz to Nyquist into `n_bands` bands spaced
    /// evenly on the Bark scale. Bark bands closely model the ear's critical
    /// bands and are used by MPEG psychoacoustic models.
    ///
    /// `n_bins` is the number of spectral bins in the caller's frequency
    /// representation (e.g. `MdctParams::n_coefficients()` for MDCT, or
    /// `n_fft / 2 + 1` for a real FFT). Bin indices in the returned bands
    /// index into that array.
    ///
    /// # Arguments
    /// - `n_bands` – Number of bands. 24 covers the full audible range.
    /// - `sample_rate_hz` – Sample rate of the audio signal in Hz.
    /// - `n_bins` – Total number of spectral bins.
    ///
    /// # Returns
    /// A `BandLayout` with `n_bands` entries.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # #[cfg(feature = "psychoacoustic")] {
    /// use audio_samples::BandLayout;
    /// use std::num::NonZeroUsize;
    ///
    /// let layout = BandLayout::bark(
    ///     NonZeroUsize::new(24).unwrap(),
    ///     44100.0,
    ///     NonZeroUsize::new(1024).unwrap(),
    /// );
    /// assert_eq!(layout.len().get(), 24);
    /// // All bands fit within the bin range.
    /// for band in layout.as_slice().iter() {
    ///     assert!(band.end_bin <= 1024);
    ///     assert!(band.end_bin > band.start_bin);
    /// }
    /// # }
    /// ```
    #[must_use]
    pub fn bark(n_bands: NonZeroUsize, sample_rate_hz: f32, n_bins: NonZeroUsize) -> Self {
        let n = n_bands.get();
        let bins = n_bins.get();
        let nyquist = sample_rate_hz / 2.0;
        let max_bark = hz_to_bark(nyquist);

        let bands: Vec<Band> = (0..n)
            .map(|i| {
                let bark_start = i as f32 * max_bark / n as f32;
                let bark_end = (i + 1) as f32 * max_bark / n as f32;
                let bark_centre = (bark_start + bark_end) / 2.0;

                let hz_start = bark_to_hz(bark_start);
                let hz_end = bark_to_hz(bark_end);
                let hz_centre = bark_to_hz(bark_centre);

                let start_bin = hz_to_bin(hz_start, sample_rate_hz, bins);
                let end_bin_raw = hz_to_bin(hz_end, sample_rate_hz, bins);
                // Guarantee end_bin > start_bin per Band invariant.
                let end_bin = end_bin_raw.max(start_bin + 1).min(bins);

                let perceptual_position = bark_centre / max_bark;

                // SAFETY: end_bin > start_bin is enforced above.
                unsafe { Band::new(start_bin, end_bin, hz_centre, perceptual_position) }
            })
            .collect();

        // SAFETY: n_bands >= 1 guarantees at least one element.
        let bands_non_empty = unsafe { NonEmptySlice::new_unchecked(&bands) };
        Self::new(bands_non_empty)
    }

    /// Creates a Mel-scale band layout.
    ///
    /// Divides the spectrum from 0 Hz to Nyquist into `n_bands` bands spaced
    /// evenly on the Mel scale. Mel bands are commonly used for speech and music
    /// feature extraction.
    ///
    /// `n_bins` is the number of spectral bins in the caller's frequency
    /// representation. Bin indices in the returned bands index into that array.
    ///
    /// # Arguments
    /// - `n_bands` – Number of bands. 40–128 is typical for speech / music.
    /// - `sample_rate_hz` – Sample rate of the audio signal in Hz.
    /// - `n_bins` – Total number of spectral bins.
    ///
    /// # Returns
    /// A `BandLayout` with `n_bands` entries.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # #[cfg(feature = "psychoacoustic")] {
    /// use audio_samples::BandLayout;
    /// use std::num::NonZeroUsize;
    ///
    /// let layout = BandLayout::mel(
    ///     NonZeroUsize::new(40).unwrap(),
    ///     22050.0,
    ///     NonZeroUsize::new(512).unwrap(),
    /// );
    /// assert_eq!(layout.len().get(), 40);
    /// # }
    /// ```
    #[must_use]
    pub fn mel(n_bands: NonZeroUsize, sample_rate_hz: f32, n_bins: NonZeroUsize) -> Self {
        let n = n_bands.get();
        let bins = n_bins.get();
        let nyquist = sample_rate_hz / 2.0;
        let max_mel = hz_to_mel(nyquist);

        let bands: Vec<Band> = (0..n)
            .map(|i| {
                let mel_start = i as f32 * max_mel / n as f32;
                let mel_end = (i + 1) as f32 * max_mel / n as f32;
                let mel_centre = (mel_start + mel_end) / 2.0;

                let hz_start = mel_to_hz(mel_start);
                let hz_end = mel_to_hz(mel_end);
                let hz_centre = mel_to_hz(mel_centre);

                let start_bin = hz_to_bin(hz_start, sample_rate_hz, bins);
                let end_bin_raw = hz_to_bin(hz_end, sample_rate_hz, bins);
                let end_bin = end_bin_raw.max(start_bin + 1).min(bins);

                let perceptual_position = mel_centre / max_mel;

                // SAFETY: end_bin > start_bin is enforced above.
                unsafe { Band::new(start_bin, end_bin, hz_centre, perceptual_position) }
            })
            .collect();

        // SAFETY: n_bands >= 1 guarantees at least one element.
        let bands_non_empty = unsafe { NonEmptySlice::new_unchecked(&bands) };
        Self::new(bands_non_empty)
    }

    /// Creates a CELT-style band layout using the Opus RFC 6716 band edges.
    ///
    /// Maps the 21 perceptual bands defined in RFC 6716 §4.3.1 onto the caller's
    /// spectral-bin array. Bands whose lower edge falls at or above Nyquist are
    /// omitted, so the returned layout will have **fewer than 21 bands** for
    /// sample rates below 40 kHz.
    ///
    /// This band layout is the recommended choice for the CELT mode of
    /// [`crate::codecs::opus::OpusCodec`].
    ///
    /// `n_bins` is the number of spectral bins (typically `window_size / 2`).
    ///
    /// # Arguments
    /// - `sample_rate_hz` – Sample rate of the audio signal in Hz.
    /// - `n_bins` – Total number of spectral bins.
    ///
    /// # Returns
    /// A `BandLayout` with up to 21 entries.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # #[cfg(feature = "psychoacoustic")] {
    /// use audio_samples::BandLayout;
    /// use std::num::NonZeroUsize;
    ///
    /// // 20 ms frame at 44.1 kHz → 882 samples → 441 bins
    /// let layout = BandLayout::celt(44100.0, NonZeroUsize::new(441).unwrap());
    /// assert!(layout.len().get() <= 21);
    /// for band in layout.as_slice().iter() {
    ///     assert!(band.end_bin > band.start_bin);
    /// }
    /// # }
    /// ```
    #[must_use]
    pub fn celt(sample_rate_hz: f32, n_bins: NonZeroUsize) -> Self {
        /// CELT band boundary frequencies in Hz (RFC 6716 §4.3.1, Table 2).
        ///
        /// 22 edge values define 21 bands:
        /// `[0–200), [200–400), …, [15600–20000)`.
        const CELT_BAND_EDGES_HZ: &[f32] = &[
            0.0, 200.0, 400.0, 600.0, 800.0, 1000.0, 1200.0, 1400.0, 1600.0,
            2000.0, 2400.0, 2800.0, 3200.0, 4000.0, 4800.0, 5600.0, 6800.0,
            8000.0, 9600.0, 12000.0, 15600.0, 20000.0,
        ];

        let nyquist = sample_rate_hz / 2.0;
        let bins = n_bins.get();
        let n_edges = CELT_BAND_EDGES_HZ.len();

        let mut bands: Vec<Band> = Vec::with_capacity(n_edges - 1);

        for i in 0..n_edges - 1 {
            let low_hz = CELT_BAND_EDGES_HZ[i];
            if low_hz >= nyquist {
                break;
            }
            let high_hz = CELT_BAND_EDGES_HZ[i + 1].min(nyquist);
            let centre_hz = (low_hz + high_hz) / 2.0;

            let start_bin = hz_to_bin(low_hz, sample_rate_hz, bins);
            let end_bin_raw = hz_to_bin(high_hz, sample_rate_hz, bins);
            let end_bin = end_bin_raw.max(start_bin + 1).min(bins);

            // Normalised position: linear index in [0, 1].
            let perceptual_position = i as f32 / (n_edges - 2) as f32;

            // SAFETY: end_bin > start_bin is enforced by .max(start_bin + 1) above.
            bands.push(unsafe { Band::new(start_bin, end_bin, centre_hz, perceptual_position) });
        }

        // Guarantee at least one band even for very low sample rates.
        if bands.is_empty() {
            // SAFETY: end_bin = 1 > 0 = start_bin.
            bands.push(unsafe { Band::new(0, 1, nyquist / 2.0, 0.0) });
        }

        // SAFETY: bands is non-empty (guarded by the fallback above).
        let bands_ne = unsafe { NonEmptySlice::new_unchecked(&bands) };
        Self::new(bands_ne)
    }

    /// Creates an ERB-scale band layout.
    ///
    /// Divides the spectrum from 0 Hz to Nyquist into `n_bands` bands spaced
    /// evenly on the ERB (Equivalent Rectangular Bandwidth) scale of Glasberg &
    /// Moore (1990). ERB bands more accurately model auditory filter widths than
    /// Bark bands, particularly at low frequencies, and are used in modern
    /// perceptual models (Opus, EVS).
    ///
    /// `n_bins` is the number of spectral bins in the caller's frequency
    /// representation. Bin indices in the returned bands index into that array.
    ///
    /// # Arguments
    /// - `n_bands` – Number of bands. 32–64 is typical.
    /// - `sample_rate_hz` – Sample rate of the audio signal in Hz.
    /// - `n_bins` – Total number of spectral bins.
    ///
    /// # Returns
    /// A `BandLayout` with `n_bands` entries.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # #[cfg(feature = "psychoacoustic")] {
    /// use audio_samples::BandLayout;
    /// use std::num::NonZeroUsize;
    ///
    /// let layout = BandLayout::erb(
    ///     NonZeroUsize::new(32).unwrap(),
    ///     44100.0,
    ///     NonZeroUsize::new(1024).unwrap(),
    /// );
    /// assert_eq!(layout.len().get(), 32);
    /// for band in layout.as_slice().iter() {
    ///     assert!(band.end_bin > band.start_bin);
    /// }
    /// # }
    /// ```
    #[must_use]
    pub fn erb(n_bands: NonZeroUsize, sample_rate_hz: f32, n_bins: NonZeroUsize) -> Self {
        let n = n_bands.get();
        let bins = n_bins.get();
        let nyquist = sample_rate_hz / 2.0;
        let erb_min = hz_to_erb(0.0);
        let erb_max = hz_to_erb(nyquist);
        let erb_range = erb_max - erb_min;

        let bands: Vec<Band> = (0..n)
            .map(|i| {
                let erb_start = erb_min + i as f32 * erb_range / n as f32;
                let erb_end = erb_min + (i + 1) as f32 * erb_range / n as f32;
                let erb_centre = (erb_start + erb_end) / 2.0;

                let hz_start = erb_to_hz(erb_start);
                let hz_end = erb_to_hz(erb_end);
                let hz_centre = erb_to_hz(erb_centre);

                let start_bin = hz_to_bin(hz_start, sample_rate_hz, bins);
                let end_bin_raw = hz_to_bin(hz_end, sample_rate_hz, bins);
                let end_bin = end_bin_raw.max(start_bin + 1).min(bins);

                let perceptual_position = (erb_centre - erb_min) / erb_range;

                // SAFETY: end_bin > start_bin is enforced above.
                unsafe { Band::new(start_bin, end_bin, hz_centre, perceptual_position) }
            })
            .collect();

        // SAFETY: n_bands >= 1 guarantees at least one element.
        let bands_non_empty = unsafe { NonEmptySlice::new_unchecked(&bands) };
        Self::new(bands_non_empty)
    }
}
