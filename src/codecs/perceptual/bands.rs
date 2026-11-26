//! Perceptual band layouts and the frequency-scale conversions behind them.
//!
//! A [`BandLayout`] partitions the bins of a spectrum (MDCT or FFT, whichever
//! length is passed as `n_bins`) into bands that follow the ear's frequency
//! resolution:
//!
//! - [`BandLayout::bark`]: bands of equal width on the Bark scale (Traunmüller,
//!   1990), the basis of most MPEG psychoacoustic models.
//! - [`BandLayout::mel`]: bands of equal width on the HTK (O'Shaughnessy) mel scale.
//! - [`BandLayout::erb`]: bands of equal width on the ERB-number scale
//!   (Glasberg & Moore, 1990).
//! - [`BandLayout::celt`]: the fixed band edges of the CELT codec.
//!
//! Each [`Band`] produced by these constructors stores:
//! - `start_bin` / `end_bin` - indices into the caller's frequency-bin array
//!   (MDCT or FFT, whichever was passed as `n_bins`).
//! - `centre_frequency` - centre frequency in Hz.
//! - `perceptual_position` - normalised position in [0, 1] on the respective scale.
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

/// Converts a frequency in Hz to its Bark-scale value.
///
/// Uses the Traunmüller (1990) formula:
/// `bark = 26.81 * hz / (1960 + hz) - 0.53`
///
/// This is the uncorrected formula: Traunmüller's low-end (< 2 Bark) and high-end
/// (> 20.1 Bark) corrections are not applied. Because the result is clamped at 0 Bark,
/// frequencies below about 39.5 Hz all map to 0 and do not round-trip through
/// [`bark_to_hz`].
///
/// # Arguments
/// - `hz` - Frequency in hertz. Values ≤ 0 are clamped to 0 Hz before conversion.
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
/// Inverse of the Traunmüller (1990) formula: `hz = 1960 (bark + 0.53) / (26.28 − bark)`
/// (26.28 = 26.81 − 0.53).
///
/// # Arguments
/// - `bark` - Bark value. Values < 0 are clamped to 0 before conversion.
///
/// # Returns
/// Frequency in hertz, clamped to a minimum of 0.0 Hz.
#[inline]
#[must_use]
pub fn bark_to_hz(bark: f32) -> f32 {
    let bark = bark.max(0.0);
    (1960.0_f32 * (bark + 0.53_f32) / (26.28_f32 - bark)).max(0.0)
}

/// Converts a frequency in Hz to its Mel-scale value.
///
/// Uses the O'Shaughnessy (HTK) formula: `mel = 2595 × log10(1 + hz / 700)`. This is an
/// `f32` duplicate of [`crate::utils::audio_math::hz_to_mel`]. The crate's mel
/// spectrogram (`AudioTransforms::mel_spectrogram`) uses the Slaney mel scale instead.
///
/// # Arguments
/// - `hz` - Frequency in hertz. Values ≤ 0 are clamped to 0 Hz.
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
/// Inverse of the O'Shaughnessy formula: `hz = 700 × (10^(mel / 2595) - 1)`.
///
/// # Arguments
/// - `mel` - Mel value. Values < 0 are clamped to 0.
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

/// Maps a frequency in Hz to the nearest integer bin index.
///
/// Assumes bin k lies at k·(fs/2)/(`n_bins` − 1), i.e. the real-FFT grid from 0 Hz
/// (bin 0) to Nyquist (bin `n_bins - 1`). For MDCT coefficients (N = `n_bins`), bin k
/// is actually centred at (k + ½)·fs/(2N), so the mapping is off by about half to one
/// bin. The result is clamped to `[0, n_bins - 1]`.
#[inline]
fn hz_to_bin(hz: f32, sample_rate_hz: f32, n_bins: usize) -> usize {
    let nyquist = sample_rate_hz / 2.0;
    let bin = (hz / nyquist * (n_bins - 1) as f32).round() as isize;
    bin.clamp(0, (n_bins - 1) as isize) as usize
}

/// Returns the equivalent rectangular **bandwidth** (in Hz) of the auditory filter
/// centred at `hz` (Glasberg & Moore, 1990).
///
/// `ERB(f) = 24.7 × (4.37 × f/1000 + 1)` Hz
///
/// At 0 Hz the minimum value is 24.7 Hz. This is a bandwidth, **not** a perceptual
/// frequency coordinate: because it is affine in Hz, spacing filters uniformly in this
/// quantity is the same as spacing them uniformly in Hz. The perceptual coordinate is
/// the ERB-number (ERB-rate) scale, [`hz_to_erb_number`].
///
/// # Arguments
/// - `hz` - Frequency in hertz. Negative values are clamped to 0.
///
/// # Returns
/// ERB bandwidth in Hz (always ≥ 24.7).
#[inline]
#[must_use]
pub fn hz_to_erb(hz: f32) -> f32 {
    24.7_f32 * (4.37_f32 * hz.max(0.0) / 1000.0_f32 + 1.0_f32)
}

/// Converts an ERB bandwidth (Hz) back to the centre frequency (Hz) whose auditory filter
/// has that bandwidth.
///
/// Inverse of the Glasberg & Moore (1990) bandwidth formula:
/// `f = 1000 × (erb/24.7 - 1) / 4.37`
///
/// # Arguments
/// - `erb` - ERB-bandwidth value. Values below 24.7 (the minimum, corresponding to 0 Hz)
///   are clamped to produce 0 Hz.
///
/// # Returns
/// Frequency in hertz, clamped to a minimum of 0.0.
#[inline]
#[must_use]
pub fn erb_to_hz(erb: f32) -> f32 {
    (1000.0_f32 * (erb / 24.7_f32 - 1.0_f32) / 4.37_f32).max(0.0)
}

/// Converts a frequency in Hz to its ERB number (ERB-rate), in Cam (Glasberg & Moore, 1990).
///
/// `E(f) = 21.4 × log10(1 + 0.00437 × f)` Cam
///
/// The ERB number counts how many auditory-filter bandwidths ([`hz_to_erb`]) lie below
/// `f`: one Cam is one equivalent rectangular bandwidth. Equal steps on this scale are
/// the perceptual spacing that [`BandLayout::erb`] uses. 1 kHz is about 15.62 Cam.
///
/// # Arguments
/// - `hz` - Frequency in hertz. Negative values are clamped to 0.
///
/// # Returns
/// ERB number in Cam, 0 at 0 Hz.
#[inline]
#[must_use]
pub fn hz_to_erb_number(hz: f32) -> f32 {
    21.4_f32 * 0.00437_f32.mul_add(hz.max(0.0), 1.0).log10()
}

/// Converts an ERB number (Cam) back to a frequency in Hz.
///
/// Inverse of [`hz_to_erb_number`]: `f = (10^(E / 21.4) - 1) / 0.00437`.
///
/// # Arguments
/// - `erb_number` - ERB number in Cam. Negative values are clamped to 0.
///
/// # Returns
/// Frequency in hertz, 0 at 0 Cam.
#[inline]
#[must_use]
pub fn erb_number_to_hz(erb_number: f32) -> f32 {
    (10.0_f32.powf(erb_number.max(0.0) / 21.4_f32) - 1.0_f32) / 0.00437_f32
}

/// Scales a [`BandLayout`] from one spectral-bin count to another.
///
/// Each band's `start_bin` and `end_bin` are scaled proportionally from
/// `from_n_bins` to `to_n_bins`, preserving the frequency mapping while
/// adapting to a different MDCT window size. This is used when window
/// switching re-analyses a transient region with a shorter window that
/// produces fewer spectral bins.
///
/// # Arguments
/// - `layout` - Layout to scale.
/// - `from_n_bins` - Bin count the layout was built for.
/// - `to_n_bins` - Target bin count.
///
/// # Returns
/// A new [`BandLayout`] with the same number of bands and scaled bin ranges.
#[must_use]
pub fn scale_band_layout(
    layout: &BandLayout,
    from_n_bins: NonZeroUsize,
    to_n_bins: NonZeroUsize,
) -> BandLayout {
    let from = from_n_bins.get();
    let to = to_n_bins.get();

    let bands: Vec<super::Band> = layout
        .as_slice()
        .iter()
        .map(|band| {
            let start = band.start_bin * to / from;
            let end = (band.end_bin * to / from).max(start + 1).min(to);
            // SAFETY: end > start is enforced by the .max(start + 1) above.
            unsafe { super::Band::new(start, end, band.centre_frequency, band.perceptual_position) }
        })
        .collect();

    // SAFETY: layout is non-empty (BandLayout invariant), so bands is non-empty.
    let ne = unsafe { NonEmptySlice::new_unchecked(&bands) };
    BandLayout::new(ne)
}

impl BandLayout {
    /// Creates a Bark-scale band layout.
    ///
    /// Divides the spectrum from 0 Hz to Nyquist into `n_bands` bands spaced
    /// evenly on the Bark scale (uncorrected Traunmüller formula, see
    /// [`hz_to_bark`]). Bark bands approximate the ear's critical bands and are
    /// used by MPEG psychoacoustic models. Band edges are mapped to bins on the
    /// real-FFT grid k·(fs/2)/(`n_bins` − 1) (about half a bin off for MDCT bins).
    ///
    /// `n_bins` is the number of spectral bins in the caller's frequency
    /// representation (e.g. `MdctParams::n_coefficients()` for MDCT, or
    /// `n_fft / 2 + 1` for a real FFT). Bin indices in the returned bands
    /// index into that array.
    ///
    /// # Arguments
    /// - `n_bands` - Number of bands. 24 covers the full audible range.
    /// - `sample_rate_hz` - Sample rate of the audio signal in Hz.
    /// - `n_bins` - Total number of spectral bins.
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
    /// evenly on the HTK mel scale (see [`hz_to_mel`]). Mel bands are commonly used
    /// for speech and music feature extraction.
    ///
    /// `n_bins` is the number of spectral bins in the caller's frequency
    /// representation. Bin indices in the returned bands index into that array.
    ///
    /// # Arguments
    /// - `n_bands` - Number of bands. 40-128 is typical for speech / music.
    /// - `sample_rate_hz` - Sample rate of the audio signal in Hz.
    /// - `n_bins` - Total number of spectral bins.
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

    /// Creates an ERB-scale band layout.
    ///
    /// Divides the spectrum from 0 Hz to Nyquist into `n_bands` bands of equal width on
    /// the ERB-number scale E(f) = 21.4·log10(1 + 0.00437 f) Cam (Glasberg & Moore 1990,
    /// see [`hz_to_erb_number`]), so the bands are narrow at low frequencies and wide at
    /// high frequencies, following the bandwidths of the auditory filters. Band edges are
    /// mapped to bins on the real-FFT grid k·(fs/2)/(`n_bins` − 1) (about half a bin off
    /// for MDCT bins).
    ///
    /// The bands tile the bins `0..n_bins` without gaps or overlaps. When `n_bins` is too
    /// small to give every band at least one bin, the edges of neighbouring bands round
    /// to the same bin; each band that would be empty is merged into the band above it,
    /// so the layout then has fewer than `n_bands` bands. Use [`BandLayout::len`] for the
    /// number of bands (for example to size the perceptual weights of a
    /// `PsychoacousticConfig`).
    ///
    /// `n_bins` is the number of spectral bins in the caller's frequency
    /// representation. Bin indices in the returned bands index into that array.
    ///
    /// # Arguments
    /// - `n_bands` - Number of bands. 32-64 is typical.
    /// - `sample_rate_hz` - Sample rate of the audio signal in Hz.
    /// - `n_bins` - Total number of spectral bins.
    ///
    /// # Returns
    /// A `BandLayout` with at most `n_bands` entries; exactly `n_bands` when every band
    /// spans at least one bin.
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
    /// let bands = layout.as_slice();
    /// for band in bands.iter() {
    ///     assert!(band.end_bin > band.start_bin);
    /// }
    /// // Narrow bands at low frequencies, wide bands at high frequencies.
    /// let width = |i: usize| bands[i].end_bin - bands[i].start_bin;
    /// assert!(width(31) > 10 * width(1));
    ///
    /// // Four bins cannot hold 32 bands: empty bands are merged.
    /// let small = BandLayout::erb(
    ///     NonZeroUsize::new(32).unwrap(),
    ///     44100.0,
    ///     NonZeroUsize::new(4).unwrap(),
    /// );
    /// assert!(small.len().get() <= 4);
    /// # }
    /// ```
    #[must_use]
    pub fn erb(n_bands: NonZeroUsize, sample_rate_hz: f32, n_bins: NonZeroUsize) -> Self {
        let n = n_bands.get();
        let bins = n_bins.get();
        let nyquist = sample_rate_hz / 2.0;
        let erb_max = hz_to_erb_number(nyquist);
        let erb_edge = |i: usize| i as f32 * erb_max / n as f32;

        // Band `i` covers [edges[i], edges[i + 1]). The first edge is bin 0 and the last
        // is `bins`, so the bands tile every bin; the interior edges are rounded to bins
        // and are non-decreasing because the ERB-number scale is increasing.
        let bin_edge = |i: usize| match i {
            0 => 0,
            i if i == n => bins,
            i => hz_to_bin(erb_number_to_hz(erb_edge(i)), sample_rate_hz, bins),
        };

        let mut bands: Vec<Band> = Vec::with_capacity(n.min(bins));
        let mut start = 0;
        while start < n {
            let start_bin = bin_edge(start);
            // An empty band (its upper edge on the same bin as its lower edge) is merged
            // with the bands above it until the upper edge passes `start_bin`.
            let mut end = start + 1;
            while end < n && bin_edge(end) <= start_bin {
                end += 1;
            }
            let end_bin = bin_edge(end);
            let erb_centre = (erb_edge(start) + erb_edge(end)) / 2.0;
            let perceptual_position = if erb_max > 0.0 {
                erb_centre / erb_max
            } else {
                0.0
            };
            // SAFETY: the inner loop stops at an edge above `start_bin`, or at the last
            // edge, `bins`, which is above every interior edge (at most `bins - 1`).
            bands.push(unsafe {
                Band::new(
                    start_bin,
                    end_bin,
                    erb_number_to_hz(erb_centre),
                    perceptual_position,
                )
            });
            start = end;
        }

        // SAFETY: n_bands >= 1, so the loop runs at least once and pushes a band.
        let bands_non_empty = unsafe { NonEmptySlice::new_unchecked(&bands) };
        Self::new(bands_non_empty)
    }

    /// Creates a CELT-scale band layout using the RFC 6716 band edges.
    ///
    /// Divides the spectrum into up to 21 perceptual bands whose boundaries
    /// follow the CELT band edges of RFC 6716 (§4.3, Table 55 "MDCT Bands"). Bands that
    /// fall entirely above the Nyquist frequency of `sample_rate_hz` are
    /// omitted, so the returned layout may have fewer than 21 entries.
    ///
    /// # Arguments
    /// - `sample_rate_hz` - Sample rate of the signal in Hz.
    /// - `n_bins` - Number of spectral bins (half the FFT size).
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
        /// CELT band boundary frequencies in Hz (RFC 6716 §4.3, Table 55).
        ///
        /// 22 edge values define 21 bands:
        /// `[0-200), [200-400), …, [15600-20000)`.
        const CELT_BAND_EDGES_HZ: &[f32] = &[
            0.0, 200.0, 400.0, 600.0, 800.0, 1000.0, 1200.0, 1400.0, 1600.0, 2000.0, 2400.0,
            2800.0, 3200.0, 4000.0, 4800.0, 5600.0, 6800.0, 8000.0, 9600.0, 12000.0, 15600.0,
            20000.0,
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

            let perceptual_position = i as f32 / (n_edges - 2) as f32;

            // SAFETY: end_bin > start_bin is enforced by .max(start_bin + 1) above.
            bands.push(unsafe { Band::new(start_bin, end_bin, centre_hz, perceptual_position) });
        }

        if bands.is_empty() {
            // SAFETY: end_bin = 1 > 0 = start_bin.
            bands.push(unsafe { Band::new(0, 1, nyquist / 2.0, 0.0) });
        }

        let bands_ne = NonEmptySlice::new(&bands).expect("bands is non-empty");
        Self::new(bands_ne)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nz(n: usize) -> NonZeroUsize {
        NonZeroUsize::new(n).unwrap()
    }

    /// Checks that the bands tile `0..n_bins` in order, each with at least one bin.
    fn assert_tiles(layout: &BandLayout, n_bins: usize) {
        let bands = layout.as_slice();
        assert_eq!(bands[0].start_bin, 0);
        assert_eq!(bands[bands.len().get() - 1].end_bin, n_bins);
        for band in bands.iter() {
            assert!(band.end_bin > band.start_bin, "empty band {band:?}");
        }
        for pair in bands.as_slice().windows(2) {
            assert_eq!(pair[0].end_bin, pair[1].start_bin);
            assert!(pair[1].centre_frequency > pair[0].centre_frequency);
            assert!(pair[1].perceptual_position > pair[0].perceptual_position);
        }
    }

    #[test]
    fn erb_number_matches_glasberg_moore_and_round_trips() {
        assert_eq!(hz_to_erb_number(0.0), 0.0);
        assert_eq!(hz_to_erb_number(-5.0), 0.0);
        assert!((hz_to_erb_number(1_000.0) - 15.62).abs() < 0.01);
        for f in [50.0f32, 440.0, 1_000.0, 4_000.0, 16_000.0] {
            let expected = 21.4 * (1.0 + 0.00437 * f64::from(f)).log10();
            assert!((f64::from(hz_to_erb_number(f)) - expected).abs() < 1e-4);
            assert!((erb_number_to_hz(hz_to_erb_number(f)) - f).abs() / f < 1e-4);
        }
        assert_eq!(erb_number_to_hz(-1.0), 0.0);
    }

    #[test]
    fn erb_bands_are_equally_spaced_in_erb_number() {
        let (n_bands, sample_rate, n_bins) = (32, 44_100.0f32, 4096);
        let layout = BandLayout::erb(nz(n_bands), sample_rate, nz(n_bins));
        assert_eq!(layout.len().get(), n_bands);
        assert_tiles(&layout, n_bins);

        let erb_max = hz_to_erb_number(sample_rate / 2.0);
        let step = erb_max / n_bands as f32;
        for (i, band) in layout.as_slice().iter().enumerate() {
            // Centres sit half a step into each band.
            let centre = hz_to_erb_number(band.centre_frequency);
            assert!(
                (centre - (i as f32 + 0.5) * step).abs() < 1e-3,
                "band {i}: {centre}"
            );
            assert!((band.perceptual_position - (i as f32 + 0.5) / n_bands as f32).abs() < 1e-5);
            // The lower edge is the bin of the frequency `i` steps up the ERB-number scale.
            let edge_hz = erb_number_to_hz(i as f32 * step);
            let expected = (edge_hz / (sample_rate / 2.0) * (n_bins - 1) as f32).round() as usize;
            assert_eq!(band.start_bin, expected, "band {i}");
        }

        // ERB spacing is not linear: the bands widen with frequency.
        let bands = layout.as_slice();
        let widths: Vec<usize> = bands.iter().map(|b| b.end_bin - b.start_bin).collect();
        assert!(widths[n_bands - 1] > 20 * widths[0]);
        let low_half: usize = widths[..n_bands / 2].iter().sum();
        assert!(low_half < n_bins / 8, "low half spans {low_half} bins");
    }

    #[test]
    fn erb_with_few_bins_merges_empty_bands() {
        for n_bins in 1..=64 {
            for n_bands in [1, 2, 8, 32, 64, 200] {
                let layout = BandLayout::erb(nz(n_bands), 48_000.0, nz(n_bins));
                assert!(layout.len().get() <= n_bands.min(n_bins));
                assert_tiles(&layout, n_bins);
            }
        }
        let one = BandLayout::erb(nz(32), 48_000.0, nz(1));
        assert_eq!(one.len().get(), 1);
        assert_eq!(
            (one.as_slice()[0].start_bin, one.as_slice()[0].end_bin),
            (0, 1)
        );
    }

    #[test]
    fn erb_with_a_degenerate_sample_rate_does_not_panic() {
        for sample_rate in [0.0f32, -8_000.0, f32::NAN, f32::INFINITY] {
            let layout = BandLayout::erb(nz(16), sample_rate, nz(128));
            let bands = layout.as_slice();
            assert_eq!(bands[0].start_bin, 0);
            assert_eq!(bands[bands.len().get() - 1].end_bin, 128);
        }
    }
}
