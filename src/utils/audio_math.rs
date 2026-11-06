//! Audio-domain mathematical utilities and canonical unit conversions.
//!
//! This module provides the conversions commonly needed when working with audio signals,
//! spectral representations, and musical abstractions: between physical units (time,
//! frequency, amplitude, power), perceptual scales (mel, decibels, cents), and symbolic
//! musical representations (note names, MIDI numbers).
//!
//! Keeping these conversions in one module gives the crate a single definition of each.
//! When unit handling is duplicated across pipelines or tools, small inconsistencies
//! accumulate; making the conversions part of the public API keeps their behaviour
//! stable and explicit.
//!
//! Most functions operate on scalar values and return plain numeric types or small domain
//! objects, for use in preprocessing, analysis, visualisation, and validation. Where a
//! conversion is reversible, its inverse is a separate function, with no shared or
//! implicit state.
//!
//! # Conventions
//!
//! - **Mel scale (HTK).** [`hz_to_mel`], [`mel_to_hz`], [`mel_scale`] and
//!   [`mel_frequencies`] use the HTK / O'Shaughnessy (1987) formula
//!   m = 2595·log10(1 + f/700), with f in Hz and m in mel (1000 Hz ≈ 1000 mel).
//!   **It is not the mel scale used by the spectrogram path:**
//!   `AudioTransforms::mel_spectrogram` and `AudioTransforms::mfcc` (via the
//!   `spectrograms` crate) use the Slaney scale (linear at 200/3 Hz per mel below
//!   1 kHz, logarithmic above; 1000 Hz = 15 mel). Do not use these helpers to label
//!   the bands of a mel spectrogram; [`hz_to_mel_with_scale`] and
//!   [`mel_frequencies_with_scale`] take the scale as an argument.
//! - **Decibels.** [`amplitude_to_db`] uses 20·log10(a) and [`power_to_db`] uses
//!   10·log10(p), both with reference 1 and no `top_db`. Non-positive inputs return
//!   the sentinel −80 dB, but positive inputs are **not** clamped, so results below
//!   −80 dB occur (e.g. `amplitude_to_db(1e-10) = -200`). For a lower clamp, a
//!   reference value and a `top_db` floor see `operations::spectrum::amplitude_to_db`
//!   and `operations::spectrum::power_to_db`.
//! - **MIDI and notes.** A4 = MIDI 69 = 440 Hz, f = 440·2^((m − 69)/12); C4 = MIDI 60.
//!
//! # Usage
//!
//! ```rust
//! use audio_samples::utils::audio_math::{
//!     hz_to_mel,
//!     mel_to_hz,
//!     amplitude_to_db,
//!     note_to_midi,
//!     midi_to_hz,
//! };
//!
//! // Frequency conversions between physical and perceptual scales.
//! let mel = hz_to_mel(440.0);
//! let hz = mel_to_hz(mel);
//!
//! // Amplitude expressed on a logarithmic scale.
//! let db = amplitude_to_db(0.5);
//!
//! // Musical representations.
//! let midi = note_to_midi("A4").unwrap();
//! let freq = midi_to_hz(midi as f64);
//! ```

use crate::{AudioSampleError, AudioSampleResult, NoteParseError, ParameterError};
use std::num::NonZeroUsize;

// =============================================================================
// FREQUENCY CONVERSIONS
// =============================================================================

/// Converts a frequency expressed in Hertz into the mel perceptual scale.
///
/// **This uses the HTK mel scale** (`2595 * log10(1 + f / 700)`). The Slaney scale gives
/// very different values (1 kHz is ~1000 HTK mels but 15 Slaney mels); use
/// [`hz_to_mel_with_scale`] with [`MelScale::Slaney`] for the Slaney scale.
///
/// This function is intended for situations where linear frequency values need to be mapped
/// into a perceptual representation, such as when constructing mel filter banks, analysing
/// spectral content in perceptual units, or comparing pitch-related quantities on a scale
/// that better reflects human sensitivity.
///
/// Uses the HTK (O'Shaughnessy 1987) formula m = 2595·log10(1 + f/700), with `f` in Hz
/// and `m` in mel.
///
/// Note: `AudioTransforms::mel_spectrogram` and `AudioTransforms::mfcc` use the Slaney
/// mel scale, so this function does not give the mel
/// coordinates of their bands (at 1000 Hz, HTK gives ≈ 1000 mel and Slaney gives 15).
///
/// The conversion is deterministic, monotonic for positive frequencies, and reversible via
/// [`mel_to_hz`] within normal floating-point precision limits.
///
/// # Arguments
///
/// * `freq_hz`\
///   The input frequency in Hertz. Values should be non-negative. Negative inputs are not
///   meaningful in the context of physical frequency and will propagate through the
///   underlying floating-point operations.
///
/// # Returns
///
/// The corresponding value on the mel scale, expressed in the same floating-point type as
/// the input.
///
/// # Behavioural Guarantees
///
/// * For `freq_hz >= 0`, the returned value is finite and non-decreasing with increasing
///   input frequency.
/// * Applying [`mel_to_hz`] to the returned value recovers the original frequency up to
///   floating-point rounding error.
///
/// # Examples
///
/// ```rust
/// use audio_samples::utils::audio_math::hz_to_mel;
///
/// let mel_1khz = hz_to_mel(1000.0);
/// let mel_a4 = hz_to_mel(440.0);
///
/// assert!(mel_1khz > mel_a4);
/// ```
#[inline]
#[must_use]
pub fn hz_to_mel(freq_hz: f64) -> f64 {
    2595.0 * (1.0 + freq_hz / 700.0).log10()
}

/// Converts a value expressed on the mel perceptual scale back into Hertz.
///
/// **This uses the HTK mel scale** (`2595 * log10(1 + f / 700)`). The Slaney scale gives
/// very different values (1 kHz is ~1000 HTK mels but 15 Slaney mels); use
/// [`mel_to_hz_with_scale`] with [`MelScale::Slaney`] for the Slaney scale.
///
/// This function performs the inverse mapping of [`hz_to_mel`]. It is intended for cases
/// where perceptual-domain representations need to be projected back into physical
/// frequency space, such as when constructing frequency bins, interpreting mel-domain
/// features, or synthesising frequency-aligned outputs.
///
/// Uses the inverse HTK formula f = 700·(10^(m/2595) − 1), with `m` in mel and `f` in Hz;
/// see [`hz_to_mel`] for the Slaney-scale caveat.
///
/// The conversion is deterministic, monotonic for non-negative inputs, and reversible via
/// [`hz_to_mel`] within normal floating-point precision limits.
///
/// # Arguments
///
/// * `mel`\
///   A value on the mel scale. Values should be non-negative. Negative inputs are not
///   meaningful in perceptual pitch space and will propagate through the underlying
///   floating-point operations.
///
/// # Returns
///
/// The corresponding frequency in Hertz, expressed in the same floating-point type as the
/// input.
///
/// # Behavioural Guarantees
///
/// * For `mel >= 0`, the returned frequency is finite and non-decreasing with increasing
///   mel values.
/// * Applying [`hz_to_mel`] to the returned value recovers the original mel input up to
///   floating-point rounding error.
///
/// # Examples
///
/// ```rust
/// use audio_samples::utils::audio_math::{hz_to_mel, mel_to_hz};
///
/// let freq = 1000.0;
/// let mel = hz_to_mel(freq);
/// let recovered = mel_to_hz(mel);
///
/// // Round-trip should recover the original frequency within floating-point precision.
/// assert!((freq - recovered).abs() < 1e-6);
/// ```
#[inline]
#[must_use]
pub fn mel_to_hz(mel: f64) -> f64 {
    700.0 * (10.0f64.powf(mel / 2595.0) - 1.0)
}

/// Generates a sequence of frequencies whose spacing is uniform in the mel domain.
///
/// **This uses the HTK mel scale** (`2595 * log10(1 + f / 700)`). The Slaney scale gives
/// very different values (1 kHz is ~1000 HTK mels but 15 Slaney mels); use
/// [`mel_frequencies_with_scale`] with [`MelScale::Slaney`] for the Slaney scale.
///
/// This function is identical to [`mel_frequencies`].
///
/// This function is typically used when sampling a frequency range in perceptual space,
/// for example when constructing mel-aligned filter banks, visualisations, or validation
/// grids. The returned values are expressed in Hertz, but their distribution corresponds
/// to equal spacing after conversion to the mel scale.
///
/// The sequence includes both endpoints and preserves ordering from low to high
/// frequency when `fmin <= fmax`.
///
/// Spacing is uniform on the HTK mel scale (see [`hz_to_mel`]). It does **not** give the
/// band centres of `AudioTransforms::mel_spectrogram`, which uses the Slaney mel scale.
///
/// # Arguments
///
/// * `n_mels`\
///   The number of frequency points to generate.\
///   If `n_mels == 0`, an empty vector is returned.\
///   If `n_mels == 1`, the returned vector contains only `fmin`.
///
/// * `fmin`\
///   The lower bound of the frequency range in Hertz. Values should be non-negative.
///
/// * `fmax`\
///   The upper bound of the frequency range in Hertz. Values should be non-negative.
///
/// # Returns
///
/// A vector of length `n_mels` containing frequencies in Hertz. The first element
/// corresponds approximately to `fmin` and the final element corresponds approximately
/// to `fmax`, subject to floating-point rounding.
///
/// # Behavioural Guarantees
///
/// * For `n_mels >= 2` and `fmin <= fmax`, the returned sequence is monotonically
///   non-decreasing.
/// * The endpoints are stable under round-trip conversion through the mel scale within
///   normal floating-point precision limits.
///
/// # Examples
///
/// ```rust
/// use audio_samples::utils::audio_math::mel_scale;
///
/// let freqs = mel_scale(10, 100.0, 8_000.0);
///
/// assert_eq!(freqs.len(), 10);
/// assert!((freqs[0] - 100.0).abs() < 1e-6);
/// assert!((freqs[9] - 8_000.0).abs() < 1e-6);
/// ```
#[inline]
#[must_use]
pub fn mel_scale(n_mels: usize, fmin: f64, fmax: f64) -> Vec<f64> {
    if n_mels == 0 {
        return Vec::new();
    }
    if n_mels == 1 {
        return vec![fmin];
    }

    let mel_min = hz_to_mel(fmin);
    let mel_max = hz_to_mel(fmax);

    let mel_step = (mel_max - mel_min) / (n_mels - 1) as f64;

    (0..n_mels)
        .map(|i| {
            let mel_val = (i as f64).mul_add(mel_step, mel_min);
            mel_to_hz(mel_val)
        })
        .collect()
}

/// Converts a frequency expressed in Hertz into a MIDI note value.
///
/// This function maps physical frequency into the MIDI pitch domain. The returned value is
/// continuous rather than quantised, allowing fractional MIDI values to represent detuning
/// between semitone boundaries. This is useful for pitch tracking, tuning analysis, and
/// continuous pitch estimation workflows.
///
/// The conversion is deterministic, monotonic for positive frequencies, and reversible via
/// [`midi_to_hz`] within normal floating-point precision limits.
///
/// # Arguments
///
/// * `freq_hz`\
///   The input frequency in Hertz. Values must be strictly positive. Zero or negative inputs
///   are not meaningful in logarithmic pitch space and will propagate through the underlying
///   floating-point operations.
///
/// # Returns
///
/// A MIDI note value expressed in the same floating-point type as the input. Integer values
/// correspond to equal-tempered semitone indices, while fractional values represent
/// sub-semitone offsets.
///
/// # Behavioural Guarantees
///
/// * For `freq_hz > 0`, the returned value is finite and strictly increasing with increasing
///   input frequency.
/// * Applying [`midi_to_hz`] to the returned value recovers the original frequency up to
///   floating-point rounding error.
///
/// # Examples
///
/// ```rust
/// use audio_samples::utils::audio_math::hz_to_midi;
///
/// let midi_a4 = hz_to_midi(440.0);
/// let midi_c4 = hz_to_midi(261.63);
///
/// assert!((midi_a4 - 69.0).abs() < 1e-6);
/// assert!((midi_c4 - 60.0).abs() < 1e-2);
/// ```
#[inline]
#[must_use]
pub fn hz_to_midi(freq_hz: f64) -> f64 {
    12.0f64.mul_add((freq_hz / 440.0).log2(), 69.0)
}

/// Converts a MIDI note value into a frequency expressed in Hertz.
///
/// This function performs the inverse mapping of [`hz_to_midi`]. Fractional MIDI values are
/// interpreted as continuous pitch offsets rather than discrete note indices, allowing
/// smooth reconstruction of detuned or interpolated pitch values.
///
/// The conversion is deterministic, monotonic, and reversible via [`hz_to_midi`] within
/// normal floating-point precision limits.
///
/// # Arguments
///
/// * `midi_note`\
///   A MIDI note value. Fractional values are permitted and represent sub-semitone offsets.
///   Extremely large or small values may overflow the underlying floating-point
///   representation.
///
/// # Returns
///
/// The corresponding frequency in Hertz, expressed in the same floating-point type as the
/// input.
///
/// # Behavioural Guarantees
///
/// * The returned frequency is strictly increasing with increasing `midi_note`.
/// * Applying [`hz_to_midi`] to the returned value recovers the original MIDI value up to
///   floating-point rounding error.
///
/// # Examples
///
/// ```rust
/// use audio_samples::utils::audio_math::midi_to_hz;
///
/// let freq_a4 = midi_to_hz(69.0);
/// let freq_c4 = midi_to_hz(60.0);
///
/// assert!((freq_a4 - 440.0).abs() < 1e-6);
/// assert!((freq_c4 - 261.63).abs() < 1e-2);
/// ```
#[inline]
#[must_use]
pub fn midi_to_hz(midi_note: f64) -> f64 {
    440.0 * ((midi_note - 69.0) / 12.0).exp2()
}

// =============================================================================
// AMPLITUDE CONVERSIONS
// =============================================================================

/// Converts a linear amplitude ratio into a logarithmic decibel representation.
///
/// This function is used when amplitude values need to be expressed on a logarithmic scale,
/// for example when visualising dynamic range, applying perceptual thresholds, or comparing
/// relative signal levels.
///
/// Computes 20·log10(`amplitude`) (reference 1, so the result is dB re 1). To avoid
/// infinite values, non-positive amplitudes return the sentinel value `-80 dB`. This is
/// **not** a floor: positive inputs are not clamped, so very small amplitudes give values
/// below −80 dB (e.g. `amplitude_to_db(1e-10) = -200`).
///
/// Note: there is no lower clamp, no reference argument and no `top_db` limit; see
/// `operations::spectrum::amplitude_to_db` for those.
///
/// # Arguments
///
/// * `amplitude`\
///   The input amplitude ratio. Values greater than zero represent valid signal magnitudes.
///   Zero or negative values are treated as silence and mapped to the −80 dB sentinel.
///
/// # Returns
///
/// The amplitude expressed in decibels. For positive inputs this is a continuous logarithmic
/// mapping. For non-positive inputs the returned value is exactly `-80 dB`.
///
/// # Behavioural Guarantees
///
/// * For `amplitude > 0`, the returned value is finite and strictly increasing with increasing
///   amplitude.
/// * For `amplitude <= 0`, the returned value is exactly `-80 dB`.
/// * Applying [`db_to_amplitude`] to a value produced by this function recovers the original
///   amplitude up to floating-point rounding error for positive inputs
///   (the sentinel is not invertible).
///
/// # Examples
///
/// ```rust
/// use audio_samples::utils::audio_math::amplitude_to_db;
///
/// let unity = amplitude_to_db(1.0);
/// let half = amplitude_to_db(0.5);
/// let silence = amplitude_to_db(0.0);
///
/// assert!((unity - 0.0).abs() < 1e-6);
/// assert!(half < 0.0);
/// assert_eq!(silence, -80.0);
/// ```
#[inline]
#[must_use]
pub fn amplitude_to_db(amplitude: f64) -> f64 {
    if amplitude > 0.0 {
        20.0 * amplitude.log10()
    } else {
        -80.0
    }
}

/// Converts a decibel value into a linear amplitude ratio.
///
/// This function performs the inverse mapping of [`amplitude_to_db`] for values that are
/// above the floor threshold. It is commonly used when reconstructing linear amplitudes
/// from logarithmic representations, such as when applying gain curves or interpreting
/// user-facing level controls.
///
/// No clamping or validation is performed on the input decibel value.
///
/// # Arguments
///
/// * `db`\
///   A value expressed in decibels. Any finite floating-point value is accepted.
///
/// # Returns
///
/// The corresponding linear amplitude ratio.
///
/// # Behavioural Guarantees
///
/// * The returned value is strictly increasing with increasing `db`.
/// * For values produced by [`amplitude_to_db`] that were not clamped to the floor, applying
///   this function recovers the original amplitude up to floating-point rounding error.
/// * Extremely large or small `db` values may overflow or underflow the underlying
///   floating-point representation.
///
/// # Examples
///
/// ```rust
/// use audio_samples::utils::audio_math::db_to_amplitude;
///
/// let unity = db_to_amplitude(0.0);
/// let attenuated = db_to_amplitude(-6.0);
///
/// assert!((unity - 1.0).abs() < 1e-6);
/// assert!(attenuated < 1.0);
/// ```
#[inline]
#[must_use]
pub fn db_to_amplitude(db: f64) -> f64 {
    10.0f64.powf(db / 20.0)
}

/// Converts a linear power ratio into a logarithmic decibel representation.
///
/// This function is used when power values need to be expressed on a logarithmic scale,
/// such as when analysing spectral energy, comparing relative signal power, or applying
/// perceptual thresholds in power-domain features.
///
/// Computes 10·log10(`power`) (reference 1, so the result is dB re 1). To avoid infinite
/// values, non-positive powers return the sentinel value `-80 dB`. This behaviour is part
/// of the public contract. It is **not** a floor: positive inputs are not clamped, so very
/// small powers give values below −80 dB (e.g. `power_to_db(1e-12) = -120`).
///
/// Note: there is no lower clamp, no reference argument and no `top_db` limit; see
/// `operations::spectrum::power_to_db` for those.
///
/// # Arguments
///
/// * `power`\
///   The input power ratio. Values greater than zero represent valid power measurements.
///   Zero or negative values are treated as silence and mapped to the −80 dB sentinel.
///
/// # Returns
///
/// The power expressed in decibels. For positive inputs this is a continuous logarithmic
/// mapping. For non-positive inputs the returned value is exactly `-80 dB`.
///
/// # Behavioural Guarantees
///
/// * For `power > 0`, the returned value is finite and strictly increasing with increasing
///   power.
/// * For `power <= 0`, the returned value is exactly `-80 dB`.
/// * Applying [`db_to_power`] to a value produced by this function recovers the original
///   power up to floating-point rounding error for positive inputs
///   (the sentinel is not invertible).
///
/// # Examples
///
/// ```rust
/// use audio_samples::utils::audio_math::power_to_db;
///
/// let unity = power_to_db(1.0);
/// let half = power_to_db(0.5);
/// let silence = power_to_db(0.0);
///
/// assert!((unity - 0.0).abs() < 1e-6);
/// assert!(half < 0.0);
/// assert_eq!(silence, -80.0);
/// ```
#[inline]
#[must_use]
pub fn power_to_db(power: f64) -> f64 {
    if power > 0.0 {
        10.0 * power.log10()
    } else {
        -80.0
    }
}

/// Converts a decibel value into a linear power ratio.
///
/// This function performs the inverse mapping of [`power_to_db`] for values that are above
/// the floor threshold. It is typically used when reconstructing linear power values from
/// logarithmic representations, such as when converting spectral features back into
/// magnitude or energy domains.
///
/// No clamping or validation is performed on the input decibel value.
///
/// # Arguments
///
/// * `db`\
///   A value expressed in decibels. Any finite floating-point value is accepted.
///
/// # Returns
///
/// The corresponding linear power ratio.
///
/// # Behavioural Guarantees
///
/// * The returned value is strictly increasing with increasing `db`.
/// * For values produced by [`power_to_db`] that were not clamped to the floor, applying this
///   function recovers the original power up to floating-point rounding error.
/// * Extremely large or small `db` values may overflow or underflow the underlying
///   floating-point representation.
///
/// # Examples
///
/// ```rust
/// use audio_samples::utils::audio_math::db_to_power;
///
/// let unity = db_to_power(0.0);
/// let attenuated = db_to_power(-3.0);
///
/// assert!((unity - 1.0).abs() < 1e-6);
/// assert!(attenuated < 1.0);
/// ```
#[inline]
#[must_use]
pub fn db_to_power(db: f64) -> f64 {
    10.0f64.powf(db / 10.0)
}

// =============================================================================
// TIME/FRAME CONVERSIONS
// =============================================================================

/// Converts a frame index into a time offset expressed in seconds.
///
/// This function maps discrete frame indices onto continuous time using a fixed hop size
/// and sample rate. It is primarily intended for aligning frame-based analyses (e.g.
/// STFT outputs, feature frames, segmentation results) with time-domain coordinates for
/// visualisation, annotation, or synchronisation.
///
/// The mapping assumes that frame `0` corresponds to time `0`, and that successive frames
/// advance by exactly `hop_size` samples.
///
/// # Arguments
///
/// * `frames`\
///   The frame index to convert. Frame indices are interpreted as non-negative offsets from
///   the start of the signal.
///
/// * `sample_rate`\
///   The sampling rate in Hertz. Must be strictly positive to produce meaningful results.
///
/// * `hop_size`\
///   The number of samples between successive frames. Must be non-zero.
///
/// # Returns
///
/// The corresponding time offset in seconds.
///
/// # Behavioural Guarantees
///
/// * The returned value is non-decreasing with increasing `frames` when `sample_rate > 0`.
/// * For fixed parameters, applying [`time_to_frames`] to the returned value recovers the
///   original frame index up to rounding error.
///
/// # Examples
///
/// ```rust
/// use audio_samples::utils::audio_math::frames_to_time;
///
/// let t0 = frames_to_time(0, 44_100.0, 512);
/// let t1 = frames_to_time(1, 44_100.0, 512);
///
/// assert!(t1 > t0);
/// ```
#[inline]
#[must_use]
pub fn frames_to_time(frames: usize, sample_rate: f64, hop_size: usize) -> f64 {
    frames as f64 * hop_size as f64 / sample_rate
}

/// Converts a time offset expressed in seconds into the nearest frame index.
///
/// This function projects continuous time coordinates back onto a discrete frame lattice
/// defined by a fixed hop size and sample rate. The result is rounded to the nearest frame
/// index rather than truncated.
///
/// # Arguments
///
/// * `time_seconds`\
///   The time offset in seconds. Negative values are permitted but will typically map to
///   zero after rounding and conversion.
///
/// * `sample_rate`\
///   The sampling rate in Hertz. Must be strictly positive to produce meaningful results.
///
/// * `hop_size`\
///   The number of samples between successive frames. Must be non-zero.
///
/// # Returns
///
/// The nearest frame index as a `usize`. The float-to-integer conversion saturates:
/// negative or NaN values give `0`, and values above `usize::MAX` give `usize::MAX`.
///
/// # Behavioural Guarantees
///
/// * For fixed parameters, applying [`frames_to_time`] to the returned frame index produces
///   a time value close to the original input, subject to rounding.
/// * Rounding is performed to the nearest integer frame index rather than truncation.
///
/// # Rounding
///
/// This function rounds to the *nearest* frame (so it round-trips [`frames_to_time`]
/// robustly) and saturates negative times to 0. [`time_to_frames_floor`] instead
/// truncates the time to a sample index and then *floors* to a frame ("which frame
/// contains this time?"), keeps negative values, and accepts an `n_fft` offset; about
/// half of all times give results one frame apart.
///
/// # Examples
///
/// ```rust
/// use audio_samples::utils::audio_math::time_to_frames;
///
/// let frame = time_to_frames(1.0, 44_100.0, 512);
///
/// assert!(frame > 0);
/// ```
#[inline]
#[must_use]
pub fn time_to_frames(time_seconds: f64, sample_rate: f64, hop_size: usize) -> usize {
    ((time_seconds * sample_rate) / hop_size as f64).round() as usize
}

/// Converts a sample index into a time offset expressed in seconds.
///
/// This function maps discrete sample indices onto continuous time using the provided
/// sampling rate. It is commonly used when aligning sample-indexed data with time-domain
/// annotations, visualisations, or external timestamps.
///
/// The mapping assumes that sample index `0` corresponds to time `0`, and that successive
/// samples are uniformly spaced according to the sampling rate.
///
/// # Arguments
///
/// * `samples`\
///   The sample index to convert. Sample indices are interpreted as non-negative offsets
///   from the start of the signal.
///
/// * `sample_rate`\
///   The sampling rate in Hertz. Must be strictly positive to produce meaningful results.
///
/// # Returns
///
/// The corresponding time offset in seconds.
///
/// # Behavioural Guarantees
///
/// * The returned value is non-decreasing with increasing `samples` when `sample_rate > 0`.
/// * For fixed parameters, applying [`seconds_to_samples`] to the returned value recovers the
///   original sample index up to rounding error.
///
/// # Examples
///
/// ```rust
/// use audio_samples::utils::audio_math::samples_to_time;
///
/// let t0 = samples_to_time(0, 44_100.0);
/// let t1 = samples_to_time(44_100, 44_100.0);
///
/// assert!(t1 > t0);
/// ```
#[inline]
#[must_use]
pub fn samples_to_time(samples: usize, sample_rate: f64) -> f64 {
    samples as f64 / sample_rate
}

/// Converts a time offset expressed in seconds into the nearest sample index.
///
/// This function projects continuous time coordinates back onto a discrete sample lattice
/// defined by a fixed sampling rate. The result is rounded to the nearest sample index
/// rather than truncated.
///
/// # Arguments
///
/// * `time_seconds`\
///   The time offset in seconds. Negative values are permitted but will typically map to
///   zero after rounding and conversion.
///
/// * `sample_rate`\
///   The sampling rate in Hertz. Must be strictly positive to produce meaningful results.
///
/// # Returns
///
/// The nearest sample index as a `usize`. The float-to-integer conversion saturates:
/// negative or NaN values give `0`, and values above `usize::MAX` give `usize::MAX`.
///
/// # Behavioural Guarantees
///
/// * For fixed parameters, applying [`samples_to_time`] to the returned sample index
///   produces a time value close to the original input, subject to rounding.
/// * Rounding is performed to the nearest integer sample index rather than truncation.
///
/// # Rounding
///
/// This function **rounds** to the nearest sample. The similarly named
/// [`crate::utils::seconds_to_samples`] **truncates** toward zero (as does
/// [`time_to_samples`]); for times that are not an exact multiple of the sample period
/// the two can differ by one sample.
///
/// # Examples
///
/// ```rust
/// use audio_samples::utils::audio_math::seconds_to_samples;
///
/// let samples = seconds_to_samples(1.0, 44_100.0);
///
/// assert_eq!(samples, 44_100);
/// ```
#[inline]
#[must_use]
pub fn seconds_to_samples(time_seconds: f64, sample_rate: f64) -> usize {
    (time_seconds * sample_rate).round() as usize
}

/// Converts a duration expressed in milliseconds into a sample count.
///
/// # Arguments
///
/// - `ms` - Duration in milliseconds.
/// - `sample_rate` - Sampling rate in Hz. Must be strictly positive.
///
/// # Returns
///
/// The number of samples corresponding to `ms` milliseconds at the given sample rate.
/// The result is rounded to the nearest integer sample.
///
/// # Examples
///
/// ```rust
/// use audio_samples::utils::audio_math::ms_to_samples;
///
/// let n = ms_to_samples(10.0, 44_100.0);
/// assert_eq!(n, 441);
/// ```
#[inline]
#[must_use]
pub fn ms_to_samples(ms: f64, sample_rate: f64) -> usize {
    (ms * 0.001 * sample_rate).round() as usize
}

// =============================================================================
// SIGNED FRAME / SAMPLE / TIME CONVERSIONS
// =============================================================================
//
// Times are truncated to samples,
// samples are floor-divided into frames, values may be negative, and an
// optional `n_fft` shifts frame positions by `n_fft / 2` samples (the frame
// centre of an un-centred STFT).

/// Frame offset in samples implied by an optional `n_fft` (`n_fft / 2`, integer halving).
#[inline]
fn n_fft_offset(n_fft: Option<usize>) -> i64 {
    n_fft.map_or(0, |n| (n / 2) as i64)
}

/// Converts a time in seconds to a sample index, truncating toward zero.
///
/// Computes `trunc(time_seconds * sample_rate)`. Unlike [`seconds_to_samples`] (which rounds
/// to the nearest sample and saturates negatives to 0) this truncates and keeps the sign.
///
/// # Arguments
///
/// * `time_seconds` - Time in seconds; may be negative.
/// * `sample_rate` - Sampling rate in Hz.
///
/// # Returns
///
/// `(time_seconds * sample_rate)` truncated toward zero. Values beyond the `i64` range
/// saturate and `NaN` maps to 0 (Rust's float-to-int cast semantics).
///
/// # Examples
///
/// ```rust
/// use audio_samples::utils::audio_math::time_to_samples;
///
/// assert_eq!(time_to_samples(0.5, 22050.0), 11025);
/// assert_eq!(time_to_samples(-0.1, 22050.0), -2205);
/// // Truncates rather than rounds: 0.99999 samples -> 0.
/// assert_eq!(time_to_samples(0.99999 / 22050.0, 22050.0), 0);
/// ```
#[inline]
#[must_use]
pub fn time_to_samples(time_seconds: f64, sample_rate: f64) -> i64 {
    (time_seconds * sample_rate) as i64
}

/// Converts a sample index to a frame index with floor division.
///
/// Computes `floor((samples - n_fft / 2) / hop_length)`, where the `n_fft / 2` offset
/// applies only when `n_fft` is `Some`. Use the offset when frames were computed without
/// centring (`centre = false`), so that a frame is identified by the sample at its centre.
///
/// # Arguments
///
/// * `samples` - Sample index; may be negative.
/// * `hop_length` - Number of samples between successive frames.
/// * `n_fft` - Optional frame length; `Some(n)` subtracts `n / 2` samples first.
///
/// # Returns
///
/// The frame index containing `samples` (negative for samples before the first frame).
///
/// # Examples
///
/// ```rust
/// use audio_samples::nzu;
/// use audio_samples::utils::audio_math::samples_to_frames;
///
/// assert_eq!(samples_to_frames(1023, nzu!(512), None), 1);
/// assert_eq!(samples_to_frames(1023, nzu!(512), Some(2048)), -1);
/// assert_eq!(samples_to_frames(-1, nzu!(512), None), -1); // floor, not truncation
/// ```
#[inline]
#[must_use]
pub fn samples_to_frames(samples: i64, hop_length: NonZeroUsize, n_fft: Option<usize>) -> i64 {
    samples
        .saturating_sub(n_fft_offset(n_fft))
        .div_euclid(hop_length.get() as i64)
}

/// Converts a frame index to the sample index where the frame starts (or is centred).
///
/// Computes `frames * hop_length + n_fft / 2`, where the offset applies only when
/// `n_fft` is `Some`.
///
/// # Arguments
///
/// * `frames` - Frame index; may be negative.
/// * `hop_length` - Number of samples between successive frames.
/// * `n_fft` - Optional frame length; `Some(n)` adds `n / 2` samples (the frame centre
///   of an un-centred STFT).
///
/// # Returns
///
/// The sample index. Arithmetic saturates at the `i64` limits instead of overflowing.
///
/// # Examples
///
/// ```rust
/// use audio_samples::nzu;
/// use audio_samples::utils::audio_math::frames_to_samples;
///
/// assert_eq!(frames_to_samples(3, nzu!(512), None), 1536);
/// assert_eq!(frames_to_samples(3, nzu!(512), Some(2048)), 2560);
/// ```
#[inline]
#[must_use]
pub fn frames_to_samples(frames: i64, hop_length: NonZeroUsize, n_fft: Option<usize>) -> i64 {
    frames
        .saturating_mul(hop_length.get() as i64)
        .saturating_add(n_fft_offset(n_fft))
}

/// Converts block indices to frame indices: `blocks * block_length`.
///
/// A block is a run of consecutive frames read together from a stream.
///
/// # Arguments
/// - `blocks`: block index; negative indices are allowed.
/// - `block_length`: frames per block.
///
/// # Returns
/// The index of the block's first frame, saturating at the limits of `i64`.
///
/// # Examples
/// ```
/// use audio_samples::nzu;
/// use audio_samples::utils::audio_math::blocks_to_frames;
///
/// assert_eq!(blocks_to_frames(3, nzu!(16)), 48);
/// ```
#[inline]
#[must_use]
pub const fn blocks_to_frames(blocks: i64, block_length: NonZeroUsize) -> i64 {
    blocks.saturating_mul(block_length.get() as i64)
}

/// Converts block indices to sample indices: `blocks * block_length * hop_length`.
///
/// # Arguments
/// - `blocks`: block index; negative indices are allowed.
/// - `block_length`: frames per block.
/// - `hop_length`: samples between frames.
///
/// # Returns
/// The index of the block's first sample, saturating at the limits of `i64`.
///
/// # Examples
/// ```
/// use audio_samples::nzu;
/// use audio_samples::utils::audio_math::blocks_to_samples;
///
/// assert_eq!(blocks_to_samples(3, nzu!(16), nzu!(512)), 24576);
/// ```
#[inline]
#[must_use]
pub const fn blocks_to_samples(
    blocks: i64,
    block_length: NonZeroUsize,
    hop_length: NonZeroUsize,
) -> i64 {
    blocks_to_frames(blocks, block_length).saturating_mul(hop_length.get() as i64)
}

/// Converts block indices to the time in seconds at which each block starts.
///
/// # Arguments
/// - `blocks`: block index; negative indices are allowed.
/// - `block_length`: frames per block.
/// - `hop_length`: samples between frames.
/// - `sample_rate`: samples per second.
///
/// # Returns
/// `blocks * block_length * hop_length / sample_rate`.
///
/// # Examples
/// ```
/// use audio_samples::nzu;
/// use audio_samples::utils::audio_math::blocks_to_time;
///
/// assert_eq!(blocks_to_time(1, nzu!(4), nzu!(100), 400.0), 1.0);
/// ```
#[inline]
#[must_use]
pub fn blocks_to_time(
    blocks: i64,
    block_length: NonZeroUsize,
    hop_length: NonZeroUsize,
    sample_rate: f64,
) -> f64 {
    blocks_to_samples(blocks, block_length, hop_length) as f64 / sample_rate
}

/// Multiplies block indices by `scale` into `out`, wrapping at the limits of `i64`.
///
/// # Arguments
/// - `blocks`: block indices.
/// - `scale`: frames per block, or samples per block.
/// - `out`: receives one index per block; must be the same length as `blocks`.
///
/// # Examples
/// ```
/// use audio_samples::utils::audio_math::blocks_scale_into;
///
/// let mut out = [0_i64; 2];
/// blocks_scale_into(&[1, 2], 16, &mut out);
/// assert_eq!(out, [16, 32]);
/// ```
pub fn blocks_scale_into(blocks: &[i64], scale: i64, out: &mut [i64]) {
    for (o, &b) in out.iter_mut().zip(blocks) {
        *o = b.wrapping_mul(scale);
    }
}

/// Converts a time in seconds to a frame index, flooring to the frame that contains it.
///
/// The time is truncated to a sample index
/// ([`time_to_samples`]) and then floor-divided into frames ([`samples_to_frames`]).
/// This answers "which frame contains this time?".
///
/// It differs from [`time_to_frames`], which rounds to the *nearest* frame and saturates
/// negative times to 0: for about half of all times the two differ by one frame.
///
/// # Arguments
///
/// * `time_seconds` - Time in seconds; may be negative.
/// * `sample_rate` - Sampling rate in Hz.
/// * `hop_length` - Number of samples between successive frames.
/// * `n_fft` - Optional frame length; `Some(n)` shifts by `n / 2` samples as in
///   [`samples_to_frames`].
///
/// # Returns
///
/// The frame index (negative for times before the first frame).
///
/// # Examples
///
/// ```rust
/// use audio_samples::nzu;
/// use audio_samples::utils::audio_math::time_to_frames_floor;
///
/// // Times 0, 0.1, ..., 0.9 s at 22050 Hz, hop 512.
/// let frames: Vec<i64> = (0..10)
///     .map(|i| time_to_frames_floor(i as f64 * 0.1, 22050.0, nzu!(512), None))
///     .collect();
/// assert_eq!(frames, vec![0, 4, 8, 12, 17, 21, 25, 30, 34, 38]);
/// assert_eq!(time_to_frames_floor(-0.1, 22050.0, nzu!(512), None), -5);
/// ```
#[inline]
#[must_use]
pub fn time_to_frames_floor(
    time_seconds: f64,
    sample_rate: f64,
    hop_length: NonZeroUsize,
    n_fft: Option<usize>,
) -> i64 {
    samples_to_frames(
        time_to_samples(time_seconds, sample_rate),
        hop_length,
        n_fft,
    )
}

/// Converts a (possibly negative) frame index to seconds, with an optional `n_fft` offset.
///
/// Computes `(frames * hop_length + n_fft / 2) / sample_rate`, the offset applying only
/// when `n_fft` is `Some`. With `n_fft = None` it agrees with [`frames_to_time`]; `Some(n)`
/// reports the time of the frame centre for an un-centred STFT (`n / 2` samples later).
///
/// # Arguments
///
/// * `frames` - Frame index; may be negative.
/// * `sample_rate` - Sampling rate in Hz.
/// * `hop_length` - Number of samples between successive frames.
/// * `n_fft` - Optional frame length for the `n / 2` sample offset.
///
/// # Returns
///
/// `frames_to_samples(frames, hop_length, n_fft) / sample_rate`, in seconds.
///
/// # Examples
///
/// ```rust
/// use audio_samples::nzu;
/// use audio_samples::utils::audio_math::frames_to_time_with_offset;
///
/// let t = frames_to_time_with_offset(10, 22050.0, nzu!(512), None);
/// assert!((t - 5120.0 / 22050.0).abs() < 1e-15);
/// let t_centre = frames_to_time_with_offset(10, 22050.0, nzu!(512), Some(2048));
/// assert!((t_centre - 6144.0 / 22050.0).abs() < 1e-15);
/// ```
#[inline]
#[must_use]
pub fn frames_to_time_with_offset(
    frames: i64,
    sample_rate: f64,
    hop_length: NonZeroUsize,
    n_fft: Option<usize>,
) -> f64 {
    frames_to_samples(frames, hop_length, n_fft) as f64 / sample_rate
}

/// Applies [`time_to_samples`] to every element of `times`.
///
/// Vectorised form of [`time_to_samples`].
///
/// # Examples
///
/// ```rust
/// use audio_samples::utils::audio_math::time_to_samples_slice;
///
/// assert_eq!(time_to_samples_slice(&[0.0, 0.5, -0.5], 100.0), vec![0, 50, -50]);
/// ```
#[must_use]
pub fn time_to_samples_slice(times: &[f64], sample_rate: f64) -> Vec<i64> {
    truncate_scaled(times, sample_rate, |s| s)
}

/// Applies [`samples_to_frames`] to every element of `samples`.
///
/// Vectorised form of [`samples_to_frames`].
///
/// # Examples
///
/// ```rust
/// use audio_samples::nzu;
/// use audio_samples::utils::audio_math::samples_to_frames_slice;
///
/// assert_eq!(samples_to_frames_slice(&[0, 511, 512, -1], nzu!(512), None), vec![0, 0, 1, -1]);
/// ```
#[must_use]
pub fn samples_to_frames_slice(
    samples: &[i64],
    hop_length: NonZeroUsize,
    n_fft: Option<usize>,
) -> Vec<i64> {
    let off = n_fft_offset(n_fft);
    let div = FloorDiv::new(hop_length.get() as i64);
    match div {
        FloorDiv::Shift(k) => samples
            .iter()
            .map(|&s| s.saturating_sub(off) >> k)
            .collect(),
        _ => samples
            .iter()
            .map(|&s| div.apply(s.saturating_sub(off)))
            .collect(),
    }
}

/// Applies [`frames_to_samples`] to every element of `frames`.
///
/// Vectorised form of [`frames_to_samples`].
///
/// # Examples
///
/// ```rust
/// use audio_samples::nzu;
/// use audio_samples::utils::audio_math::frames_to_samples_slice;
///
/// assert_eq!(frames_to_samples_slice(&[0, 1, 2], nzu!(256), Some(1024)), vec![512, 768, 1024]);
/// ```
#[must_use]
pub fn frames_to_samples_slice(
    frames: &[i64],
    hop_length: NonZeroUsize,
    n_fft: Option<usize>,
) -> Vec<i64> {
    frames
        .iter()
        .map(|&f| frames_to_samples(f, hop_length, n_fft))
        .collect()
}

/// Applies [`time_to_frames_floor`] to every element of `times`.
///
/// Vectorised form of [`time_to_frames_floor`].
///
/// # Examples
///
/// ```rust
/// use audio_samples::nzu;
/// use audio_samples::utils::audio_math::time_to_frames_floor_slice;
///
/// let f = time_to_frames_floor_slice(&[0.0, 0.1, 0.2], 22050.0, nzu!(512), None);
/// assert_eq!(f, vec![0, 4, 8]);
/// ```
#[must_use]
pub fn time_to_frames_floor_slice(
    times: &[f64],
    sample_rate: f64,
    hop_length: NonZeroUsize,
    n_fft: Option<usize>,
) -> Vec<i64> {
    let off = n_fft_offset(n_fft);
    let div = FloorDiv::new(hop_length.get() as i64);
    match div {
        FloorDiv::Shift(k) => truncate_scaled(times, sample_rate, |s| s.saturating_sub(off) >> k),
        _ => truncate_scaled(times, sample_rate, |s| div.apply(s.saturating_sub(off))),
    }
}

/// `post((t * scale) as i64)` for every `t`, where the cast truncates
/// toward zero, saturates and maps NaN to 0, exactly as [`time_to_samples`].
fn truncate_scaled(times: &[f64], scale: f64, post: impl Fn(i64) -> i64) -> Vec<i64> {
    let mut out = vec![0i64; times.len()];
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 is available (checked above).
            unsafe { truncate_scaled_avx2(times, scale, &mut out, post) };
            return out;
        }
    }
    truncate_scaled_into(times, scale, &mut out, post);
    out
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn truncate_scaled_avx2(times: &[f64], scale: f64, out: &mut [i64], post: impl Fn(i64) -> i64) {
    truncate_scaled_into(times, scale, out, post);
}

/// The body of [`truncate_scaled`]. Blocks whose products are all below
/// 2^51 in magnitude are truncated with `f64::trunc` and converted exactly
/// through the bits of `x + 1.5 * 2^52` (no scalar conversion); blocks within
/// the `i64` range use a plain truncating conversion; others take the
/// saturating cast.
#[inline(always)]
fn truncate_scaled_into(times: &[f64], scale: f64, out: &mut [i64], post: impl Fn(i64) -> i64) {
    // 1.5 * 2^52: adding it to an integer-valued |x| < 2^51 puts x in the low
    // mantissa bits.
    const MAGIC: f64 = 6_755_399_441_055_744.0;
    const SMALL: f64 = 2_251_799_813_685_248.0; // 2^51
    // 2^63: products in (-2^63, 2^63) convert without saturation.
    const LIM: f64 = 9_223_372_036_854_775_808.0;
    let (src, src_rem) = times.as_chunks::<8>();
    let (dst, dst_rem) = out.as_chunks_mut::<8>();
    for (c, o) in src.iter().zip(dst) {
        let mut p = [0.0f64; 8];
        for (b, &t) in p.iter_mut().zip(c) {
            *b = t * scale;
        }
        // NaN fails the comparisons, so blocks with NaN take the last path.
        if p.iter().all(|&v| v.abs() < SMALL) {
            for (o, &v) in o.iter_mut().zip(&p) {
                let bits = (v.trunc() + MAGIC).to_bits() as i64;
                *o = post(bits - MAGIC.to_bits() as i64);
            }
        } else if p.iter().all(|&v| v.abs() < LIM) {
            for (o, &v) in o.iter_mut().zip(&p) {
                // SAFETY: `v` is finite and strictly inside the i64 range,
                // where the unchecked conversion truncates like `as`.
                *o = post(unsafe { v.to_int_unchecked::<i64>() });
            }
        } else {
            for (o, &v) in o.iter_mut().zip(&p) {
                *o = post(v as i64);
            }
        }
    }
    for (o, &t) in dst_rem.iter_mut().zip(src_rem) {
        *o = post((t * scale) as i64);
    }
}

/// `x.div_euclid(d)` for a fixed `d`, without a hardware division for the
/// usual inputs.
///
/// A power-of-two `d > 0` is an arithmetic shift (floor division for every
/// `i64`). Otherwise non-negative values use the round-up multiplier of
/// Granlund and Montgomery ("Division by invariant integers using
/// multiplication", 1994, theorem 4.2 with N = 63): with
/// `l = ceil(log2 d)` and `m = ceil(2^(63 + l) / d)`,
/// `floor(x / d) = floor(x * m / 2^(63 + l))` for `0 <= x < 2^63`. Negative
/// values, and `d <= 0`, use `div_euclid`. The result is the same as
/// `div_euclid` for every input.
#[derive(Clone, Copy)]
enum FloorDiv {
    Shift(u32),
    Magic { d: i64, m: u64, shift: u32 },
    Plain(i64),
}

impl FloorDiv {
    fn new(d: i64) -> Self {
        if d <= 0 {
            return Self::Plain(d);
        }
        let du = d as u64;
        if du.is_power_of_two() {
            return Self::Shift(du.trailing_zeros());
        }
        // d is not a power of two, so 2^(l - 1) < d < 2^l with 2 <= l <= 63
        // and m < 2^64.
        let l = 64 - (du - 1).leading_zeros();
        let shift = 63 + l;
        let m = (1u128 << shift).div_ceil(u128::from(du));
        Self::Magic {
            d,
            m: m as u64,
            shift,
        }
    }

    #[inline]
    fn apply(self, x: i64) -> i64 {
        match self {
            Self::Shift(k) => x >> k,
            Self::Magic { d, m, shift } => {
                if x >= 0 {
                    ((u128::from(x as u64) * u128::from(m)) >> shift) as i64
                } else {
                    x.div_euclid(d)
                }
            }
            Self::Plain(d) => x.div_euclid(d),
        }
    }
}

/// Applies [`frames_to_time_with_offset`] to every element of `frames`.
///
/// Vectorised form of [`frames_to_time_with_offset`].
///
/// # Examples
///
/// ```rust
/// use audio_samples::nzu;
/// use audio_samples::utils::audio_math::frames_to_time_with_offset_slice;
///
/// let t = frames_to_time_with_offset_slice(&[0, 1], 1000.0, nzu!(100), None);
/// assert_eq!(t, vec![0.0, 0.1]);
/// ```
#[must_use]
pub fn frames_to_time_with_offset_slice(
    frames: &[i64],
    sample_rate: f64,
    hop_length: NonZeroUsize,
    n_fft: Option<usize>,
) -> Vec<f64> {
    frames
        .iter()
        .map(|&f| frames_to_time_with_offset(f, sample_rate, hop_length, n_fft))
        .collect()
}

/// Returns the time (in seconds) of each of `n_frames` consecutive frames.
///
/// This is the time axis for a frame-wise feature matrix with `n_frames` columns.
///
/// # Examples
///
/// ```rust
/// use audio_samples::nzu;
/// use audio_samples::utils::audio_math::times_like;
///
/// let t = times_like(3, 22050.0, nzu!(512), None);
/// assert_eq!(t, vec![0.0, 512.0 / 22050.0, 1024.0 / 22050.0]);
/// ```
#[must_use]
pub fn times_like(
    n_frames: usize,
    sample_rate: f64,
    hop_length: NonZeroUsize,
    n_fft: Option<usize>,
) -> Vec<f64> {
    (0..n_frames as i64)
        .map(|f| frames_to_time_with_offset(f, sample_rate, hop_length, n_fft))
        .collect()
}

/// Returns the sample index of each of `n_frames` consecutive frames.
///
/// This is the sample axis for a frame-wise feature matrix with `n_frames` columns.
///
/// # Examples
///
/// ```rust
/// use audio_samples::nzu;
/// use audio_samples::utils::audio_math::samples_like;
///
/// assert_eq!(samples_like(3, nzu!(512), None), vec![0, 512, 1024]);
/// ```
#[must_use]
pub fn samples_like(n_frames: usize, hop_length: NonZeroUsize, n_fft: Option<usize>) -> Vec<i64> {
    (0..n_frames as i64)
        .map(|f| frames_to_samples(f, hop_length, n_fft))
        .collect()
}

// =============================================================================
// MUSICAL THEORY FUNCTIONS
// =============================================================================

/// Semitone offset of each natural note letter from C.
const fn letter_semitone(letter: char) -> Option<i64> {
    match letter.to_ascii_uppercase() {
        'C' => Some(0),
        'D' => Some(2),
        'E' => Some(4),
        'F' => Some(5),
        'G' => Some(7),
        'A' => Some(9),
        'B' => Some(11),
        _ => None,
    }
}

/// Semitone offset of an accidental symbol (the set accepted by [`note_to_midi`]).
const fn accidental_semitones(symbol: char) -> Option<i64> {
    match symbol {
        '#' | '\u{266F}' => Some(1),        // # and ♯
        'b' | '!' | '\u{266D}' => Some(-1), // b, ! and ♭
        '\u{1D12A}' => Some(2),             // 𝄪 double sharp
        '\u{1D12B}' => Some(-2),            // 𝄫 double flat
        '\u{266E}' => Some(0),              // ♮ natural
        _ => None,
    }
}

/// Parses an optionally signed run of ASCII digits starting at byte `start` of
/// `text`. Returns the value and the byte index just past it, or `None` if there
/// are no digits (or the number overflows).
fn parse_signed_digits(text: &str, start: usize, sign_required: bool) -> Option<(i64, usize)> {
    let bytes = text.as_bytes();
    let mut pos = start;
    let mut negative = false;
    match bytes.get(pos) {
        Some(b'+') => pos += 1,
        Some(b'-') => {
            negative = true;
            pos += 1;
        }
        _ if sign_required => return None,
        _ => {}
    }
    let digits_start = pos;
    while bytes.get(pos).is_some_and(u8::is_ascii_digit) {
        pos += 1;
    }
    if pos == digits_start {
        return None;
    }
    let magnitude: i64 = text[digits_start..pos].parse().ok()?;
    Some((if negative { -magnitude } else { magnitude }, pos))
}

const MIDI_TO_NOTE_MAP: [&str; 12] = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
];

/// Parses a scientific pitch notation string and converts it into a MIDI note number.
///
/// This function converts human-readable musical notation
/// (e.g. `"A4"`, `"C#3"`, `"Bb2"`) to numeric MIDI pitch indices. It is intended for
/// configuration, annotation, testing, and tooling scenarios rather than high-throughput
/// parsing.
///
/// # Arguments
///
/// * `note_name`\
///   A note name encoded using scientific pitch notation with an explicit octave number:
///
///   * a pitch letter `A`-`G`, upper or lower case;
///   * any number of accidentals: `#` or `♯` (+1), `b`, `!` or `♭` (-1), `𝄪` (+2),
///     `𝄫` (-2), `♮` (0). Accidentals add up (`"C##4"` is D4) and may cross an
///     octave boundary without changing the octave number (`"B#3"` is C4 = 60,
///     `"Cb4"` is B3 = 59);
///   * an octave number, optionally signed (`"C-1"` is MIDI 0);
///   * optionally a signed cents offset (`"A4+50"`, `"C4-20"`).
///
///   Examples of valid inputs include `"A4"`, `"c#3"`, `"Bb2"`, `"E♭5"`, `"G-1"`
///   and `"F𝄪4"`.
///
/// # Returns
///
/// The corresponding MIDI note number in the inclusive range `0..=127`. A cents
/// offset is added and the result is rounded to the nearest integer, ties to even.
///
/// # Errors
///
/// * [`AudioSampleError::NoteParse`], with a
///   span pointing at the offending part of the input, if the text does not follow
///   the grammar (unknown letter, unknown accidental, missing or malformed octave,
///   malformed cents offset).
/// * [`ParameterError::OutOfRange`] if the note parses but lies outside MIDI
///   `0..=127`.
///
/// # Behavioural Guarantees
///
/// * Valid enharmonic spellings resolve to the same MIDI value.
/// * [`midi_to_note`] output always parses back to the same MIDI value.
///
/// # Limits
///
/// * The octave is required.
/// * The result is a `u8` limited to `0..=127`.
/// * `n` is not an accidental and is rejected.
///
/// # Examples
///
/// ```rust
/// use audio_samples::utils::audio_math::note_to_midi;
///
/// assert_eq!(note_to_midi("A4").unwrap(), 69);
/// assert_eq!(note_to_midi("C4").unwrap(), 60);
/// assert_eq!(note_to_midi("C#4").unwrap(), 61);
/// assert_eq!(note_to_midi("Db4").unwrap(), 61);
/// assert_eq!(note_to_midi("d\u{266D}4").unwrap(), 61); // lowercase, Unicode flat
/// assert_eq!(note_to_midi("B#3").unwrap(), 60);
/// assert_eq!(note_to_midi("C-1").unwrap(), 0);
/// assert_eq!(note_to_midi("A4+50").unwrap(), 70); // 69.5 rounds to even
/// ```
#[inline]
pub fn note_to_midi(note_name: &str) -> AudioSampleResult<u8> {
    let mut chars = note_name.char_indices().peekable();

    // Pitch letter - point the caret at it.
    let semitone = match chars.next() {
        Some((_, c)) => letter_semitone(c).ok_or_else(|| {
            NoteParseError::new(note_name, (0, c.len_utf8()), "unrecognised note")
        })?,
        None => {
            return Err(NoteParseError::new(
                note_name,
                (0, 0),
                "note name must include an octave number (e.g. `A4`)",
            )
            .into());
        }
    };

    // Accidentals.
    let mut offset = 0i64;
    while let Some(&(_, c)) = chars.peek() {
        match accidental_semitones(c) {
            Some(delta) => {
                offset += delta;
                chars.next();
            }
            None => break,
        }
    }
    let octave_start = chars.peek().map_or(note_name.len(), |&(i, _)| i);

    // Octave - point the caret at whatever follows the note token.
    if octave_start == note_name.len() {
        return Err(NoteParseError::new(
            note_name,
            (0, note_name.len()),
            "note name must include an octave number (e.g. `A4`)",
        )
        .into());
    }
    let (octave, cents_start) =
        parse_signed_digits(note_name, octave_start, false).ok_or_else(|| {
            NoteParseError::new(
                note_name,
                (octave_start, note_name.len() - octave_start),
                "invalid octave",
            )
        })?;

    // Optional cents offset; nothing may follow it.
    let cents = if cents_start == note_name.len() {
        0
    } else {
        match parse_signed_digits(note_name, cents_start, true) {
            Some((cents, end)) if end == note_name.len() => cents,
            _ => {
                return Err(NoteParseError::new(
                    note_name,
                    (cents_start, note_name.len() - cents_start),
                    "invalid cents offset; expected e.g. `+50` or `-20`",
                )
                .into());
            }
        }
    };

    // MIDI value, rounded with ties to even. Integer arithmetic is
    // exact; i64 saturating ops keep absurd octave numbers from overflowing.
    let semitones = octave
        .saturating_add(1)
        .saturating_mul(12)
        .saturating_add(semitone + offset);
    let midi = (semitones as f64 + cents as f64 / 100.0).round_ties_even();

    // Check range - the note parsed cleanly, this is a genuine range violation.
    if !(0.0..=127.0).contains(&midi) {
        return Err(AudioSampleError::Parameter(ParameterError::out_of_range(
            "note_name",
            midi,
            0,
            127,
            format!("note {note_name} maps to MIDI {midi}, outside the valid range"),
        )));
    }

    Ok(midi as u8)
}

/// Converts a MIDI note number into a scientific pitch notation string.
///
/// This function maps a numeric MIDI pitch index back into a human-readable note name with
/// an explicit octave. It is primarily intended for display, logging, reporting, and
/// debugging rather than round-tripping symbolic notation exactly.
///
/// Enharmonic spellings are normalised to sharp notation. For example, a MIDI value that
/// could be interpreted as either `"C#4"` or `"Db4"` will always be rendered as `"C#4"`.
/// Information about the original accidental is therefore not preserved.
///
/// # Arguments
///
/// * `midi_note`\
///   A MIDI note number expected to lie in the inclusive range `0..=127`.
///
/// # Returns
///
/// A note name encoded using scientific pitch notation, consisting of an uppercase pitch
/// class, an optional sharp (`#`), and a signed octave number (e.g. `"A4"`, `"C#3"`).
///
/// # Errors
///
/// Returns an error if `midi_note` lies outside the valid MIDI range `0..=127`.
///
/// # Behavioural Guarantees
///
/// * The returned string always uses sharp notation for accidentals.
/// * For MIDI 12–127, applying [`note_to_midi`] to the returned string recovers the
///   original MIDI value. MIDI 0–11 are rendered in octave −1 (`"C-1"` … `"B-1"`), which
///   [`note_to_midi`] does not accept.
///
/// # Limits
///
/// This function always writes ASCII `#` (`"C#4"`), never Unicode accidentals
/// (`"C♯4"`). Fractional MIDI values, cents output and names without an octave are
/// not supported.
///
/// # Examples
///
/// ```rust
/// use audio_samples::utils::audio_math::midi_to_note;
///
/// assert_eq!(midi_to_note(69).unwrap(), "A4");
/// assert_eq!(midi_to_note(60).unwrap(), "C4");
/// assert_eq!(midi_to_note(61).unwrap(), "C#4");
/// ```
#[inline]
pub fn midi_to_note(midi_note: u8) -> AudioSampleResult<String> {
    if midi_note > 127 {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "midi_note",
            "MIDI note must be in range 0-127",
        )));
    }

    let octave = (i32::from(midi_note) / 12) - 1;
    let note_index = midi_note % 12;
    let note_name = MIDI_TO_NOTE_MAP[note_index as usize];

    Ok(format!("{note_name}{octave}"))
}

/// [`midi_to_note`] as a `&'static str` from a table built on first use, so
/// a call allocates nothing.
///
/// # Errors
///
/// As [`midi_to_note`].
///
/// # Examples
///
/// ```rust
/// use audio_samples::utils::audio_math::{midi_to_note, midi_to_note_static};
///
/// assert_eq!(midi_to_note_static(69).unwrap(), "A4");
/// assert_eq!(midi_to_note_static(60).unwrap(), midi_to_note(60).unwrap());
/// ```
pub fn midi_to_note_static(midi_note: u8) -> AudioSampleResult<&'static str> {
    static NAMES: std::sync::LazyLock<Vec<String>> = std::sync::LazyLock::new(|| {
        (0..=127u8)
            .map(|m| midi_to_note(m).expect("0..=127 is valid"))
            .collect()
    });
    if midi_note > 127 {
        // The same error as midi_to_note.
        return Err(midi_to_note(midi_note).expect_err("out of range"));
    }
    Ok(NAMES[usize::from(midi_note)].as_str())
}

/// [`frequency_to_note`] with the note name as a `&'static str` (see
/// [`midi_to_note_static`]), so a call allocates nothing.
///
/// # Errors
///
/// As [`frequency_to_note`].
///
/// # Examples
///
/// ```rust
/// use audio_samples::utils::audio_math::{frequency_to_note, frequency_to_note_static};
///
/// let (name, cents) = frequency_to_note_static(445.0).unwrap();
/// let (owned, cents_owned) = frequency_to_note(445.0).unwrap();
/// assert_eq!((name, cents), (owned.as_str(), cents_owned));
/// ```
pub fn frequency_to_note_static(freq_hz: f64) -> AudioSampleResult<(&'static str, f64)> {
    // Invalid or out-of-range input: the same error as frequency_to_note.
    let invalid = || Err(frequency_to_note(freq_hz).expect_err("invalid frequency"));
    if !(freq_hz.is_finite() && freq_hz > 0.0) {
        return invalid();
    }
    let midi_float = hz_to_midi(freq_hz);
    let midi_rounded = midi_float.round();
    let cents = (midi_float - midi_rounded) * 100.0;
    if !(0.0..=127.0).contains(&midi_rounded) {
        return invalid();
    }
    Ok((midi_to_note_static(midi_rounded as u8)?, cents))
}

/// Converts a note name to a fractional MIDI number, keeping any cents and allowing any
/// octave.
///
/// The grammar is a pitch letter, any number of accidentals (`#`, `♯`, `𝄪`, `b`, `!`,
/// `♭`, `𝄫`, `♮`, `n`), an optional octave (0 when absent) and an optional cents suffix
/// such as `+25`. Unlike [`note_to_midi`], the result is not limited to 0..=127.
///
/// # Arguments
/// - `note_name`: the note, such as `"A#4+25"`.
///
/// # Returns
/// The MIDI number, with cents as a fraction.
///
/// # Errors
/// [`ParameterError`] if the name does not follow the grammar.
///
/// # Examples
/// ```
/// use audio_samples::utils::audio_math::note_to_midi_fractional;
///
/// assert_eq!(note_to_midi_fractional("A4+50").unwrap(), 69.5);
/// assert_eq!(note_to_midi_fractional("C").unwrap(), 12.0);
/// assert_eq!(note_to_midi_fractional("C-2").unwrap(), -12.0);
/// ```
pub fn note_to_midi_fractional(note_name: &str) -> AudioSampleResult<f64> {
    let bad = || {
        AudioSampleError::Parameter(ParameterError::invalid_value(
            "note",
            format!("improper note format: {note_name:?}"),
        ))
    };
    let mut chars = note_name.chars().peekable();
    let semitone = match chars.next().map(|c| c.to_ascii_uppercase()) {
        Some('C') => 0_i64,
        Some('D') => 2,
        Some('E') => 4,
        Some('F') => 5,
        Some('G') => 7,
        Some('A') => 9,
        Some('B') => 11,
        _ => return Err(bad()),
    };
    let mut offset = 0_i64;
    while let Some(&c) = chars.peek() {
        offset += match c {
            '#' | '♯' => 1,
            'b' | '!' | '♭' => -1,
            '𝄪' => 2,
            '𝄫' => -2,
            'n' | '♮' => 0,
            _ => break,
        };
        chars.next();
    }
    let number = |chars: &mut std::iter::Peekable<std::str::Chars<'_>>| -> Option<i64> {
        let mut text = String::new();
        if let Some(&c) = chars.peek() {
            if c == '+' || c == '-' {
                text.push(c);
                chars.next();
            }
        }
        while let Some(&c) = chars.peek() {
            if c.is_ascii_digit() {
                text.push(c);
                chars.next();
            } else {
                break;
            }
        }
        text.parse().ok()
    };
    let mut octave = 0;
    let mut cents = 0;
    let rest: String = chars.clone().collect();
    if !rest.is_empty() {
        // A leading sign without a following octave is a cents suffix only after an octave.
        octave = number(&mut chars).ok_or_else(bad)?;
        if chars.peek().is_some() {
            cents = number(&mut chars).ok_or_else(bad)?;
        }
        if chars.next().is_some() {
            return Err(bad());
        }
    }
    Ok((12 * (octave + 1) + semitone + offset) as f64 + cents as f64 * 0.01)
}

/// Converts a scientific pitch notation string into a frequency expressed in Hertz.
///
/// This is a convenience wrapper that combines [`note_to_midi`] and [`midi_to_hz`] to allow
/// direct conversion from symbolic note names into physical frequency values. It is intended
/// for configuration, testing, visualisation, and light-weight tooling rather than
/// performance-critical inner loops.
///
/// All parsing rules, validation behaviour, and error semantics are inherited from
/// [`note_to_midi`].
///
/// # Arguments
///
/// * `note_name`\
///   A note name encoded using scientific pitch notation (e.g. `"A4"`, `"C#3"`). The accepted
///   grammar and validation rules are identical to those documented in [`note_to_midi`].
///
/// # Returns
///
/// The corresponding frequency in Hertz, expressed in the requested floating-point type.
///
/// # Errors
///
/// Returns any error produced by [`note_to_midi`] if the input cannot be parsed or lies
/// outside the valid MIDI range.
///
/// # Behavioural Guarantees
///
/// * For valid inputs, the returned frequency is strictly positive.
/// * Applying [`frequency_to_note`] to the returned value yields the original note name
///   up to normalisation of enharmonic spelling and rounding.
///
/// # Examples
///
/// ```rust
/// use audio_samples::utils::audio_math::note_to_frequency;
///
/// let a4 = note_to_frequency("A4").unwrap();
/// let c4 = note_to_frequency("C4").unwrap();
///
/// assert!((a4 - 440.0).abs() < 1e-6);
/// assert!((c4 - 261.63).abs() < 1e-2);
/// ```
#[inline]
pub fn note_to_frequency(note_name: &str) -> AudioSampleResult<f64> {
    let midi_note = note_to_midi(note_name)?;
    Ok(midi_to_hz(f64::from(midi_note)))
}

/// Converts a frequency expressed in Hertz into the nearest musical note and cents offset.
///
/// This function projects a continuous frequency value onto the equal-tempered pitch grid.
/// The returned note corresponds to the nearest semitone, and the accompanying cents value
/// represents the signed deviation from that note in hundredths of a semitone.
///
/// The note name is normalised to sharp notation via [`midi_to_note`].
///
/// # Arguments
///
/// * `freq_hz`\
///   The input frequency in Hertz. Must be strictly positive.
///
/// # Returns
///
/// A tuple `(note_name, cents)` where:
///
/// * `note_name` is the nearest pitch expressed in scientific pitch notation.
/// * `cents` is the signed deviation from that pitch, where positive values indicate a
///   higher frequency and negative values indicate a lower frequency.
///
/// # Errors
///
/// * [`ParameterError::InvalidValue`] if `freq_hz` is not a finite positive number.
/// * [`ParameterError::OutOfRange`] if the nearest note lies outside MIDI `0..=127`
///   (below about 7.9 Hz or above about 12.9 kHz), consistently with [`midi_to_note`]
///   and [`note_to_midi`], which cover the same range.
///
/// # Behavioural Guarantees
///
/// * The nearest note is `round(m)` with m = 69 + 12·log2(f/440); `cents` = 100·(m − round(m))
///   and lies in `[-50, 50]`, subject to floating-point rounding.
/// * Applying [`note_to_frequency`] to the returned `note_name` yields a frequency close
///   to the nearest semitone of the original input, not necessarily the original value.
///
/// # Limits
///
/// Frequencies outside MIDI `0..=127` (octaves below C-1 or above 9) are an error. The
/// name uses ASCII accidentals and the cents are returned separately, not inside the
/// string.
///
/// # Examples
///
/// ```rust
/// use audio_samples::utils::audio_math::frequency_to_note;
///
/// let (note, cents) = frequency_to_note(440.0).unwrap();
///
/// assert_eq!(note, "A4");
/// assert!(cents.abs() < 1e-6);
///
/// // Below the MIDI range: an error, not a mislabelled "C-1".
/// assert!(frequency_to_note(4.0).is_err());
/// ```
#[inline]
pub fn frequency_to_note(freq_hz: f64) -> AudioSampleResult<(String, f64)> {
    if !(freq_hz.is_finite() && freq_hz > 0.0) {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "freq_hz",
            "Frequency must be a finite positive number",
        )));
    }

    let midi_float = hz_to_midi(freq_hz);
    let midi_rounded = midi_float.round();
    let cents = (midi_float - midi_rounded) * 100.0;

    // Range-check BEFORE converting: a saturating `as u8` cast would silently
    // turn every frequency below ~7.9 Hz into MIDI 0 ("C-1").
    if !(0.0..=127.0).contains(&midi_rounded) {
        return Err(AudioSampleError::Parameter(ParameterError::out_of_range(
            "freq_hz",
            freq_hz,
            midi_to_hz(-0.5),
            midi_to_hz(127.5),
            format!(
                "{freq_hz} Hz is nearest to MIDI {midi_rounded}, outside the representable range 0..=127"
            ),
        )));
    }

    let note_name = midi_to_note(midi_rounded as u8)?;

    Ok((note_name, cents))
}

/// Converts a pitch deviation expressed in cents into a frequency ratio.
///
/// This function maps logarithmic pitch offsets into multiplicative frequency scaling
/// factors. It is typically used when applying detuning, pitch modulation, or converting
/// musical offsets into signal-processing parameters.
///
/// A value of `0` cents corresponds to a ratio of `1.0` (no change). Positive values increase
/// frequency, while negative values decrease frequency.
///
/// # Arguments
///
/// * `cents`\
///   A pitch deviation in cents. Any finite floating-point value is accepted.
///
/// # Returns
///
/// A multiplicative frequency ratio. A value of `1.0` represents no change in frequency.
///
/// # Behavioural Guarantees
///
/// * The returned value is strictly positive for all finite inputs.
/// * The returned value is strictly increasing with increasing `cents`.
/// * Applying [`ratio_to_cents`] to the returned value recovers the original cents value
///   up to floating-point rounding error.
/// * Extremely large positive or negative inputs may overflow or underflow the underlying
///   floating-point representation.
///
/// # Examples
///
/// ```rust
/// use audio_samples::utils::audio_math::cents_to_ratio;
///
/// let octave = cents_to_ratio(1_200.0);
/// let semitone = cents_to_ratio(100.0);
///
/// assert!((octave - 2.0).abs() < 1e-6);
/// assert!(semitone > 1.0);
/// ```
#[inline]
#[must_use]
pub fn cents_to_ratio(cents: f64) -> f64 {
    (cents / 1200.0).exp2()
}

/// Converts a frequency ratio into a pitch deviation expressed in cents.
///
/// This function performs the inverse mapping of [`cents_to_ratio`]. It is commonly used
/// when interpreting frequency ratios in musical terms, such as measuring detuning or
/// estimating pitch drift.
///
/// # Arguments
///
/// * `ratio`\
///   A frequency ratio. Values must be strictly positive. Zero or negative values are not
///   meaningful in logarithmic pitch space and will propagate through the underlying
///   floating-point operations.
///
/// # Returns
///
/// The corresponding pitch deviation in cents.
///
/// # Behavioural Guarantees
///
/// * For `ratio > 0`, the returned value is finite and strictly increasing with increasing
///   ratio.
/// * Applying [`cents_to_ratio`] to the returned value recovers the original ratio up to
///   floating-point rounding error.
/// * Extremely small or large ratios may underflow or overflow the underlying
///   floating-point representation.
///
/// # Examples
///
/// ```rust
/// use audio_samples::utils::audio_math::ratio_to_cents;
///
/// let octave = ratio_to_cents(2.0);
/// let semitone = ratio_to_cents(1.059463);
///
/// assert!((octave - 1_200.0).abs() < 1e-3);
/// assert!((semitone - 100.0).abs() < 1.0);
/// ```
#[inline]
#[must_use]
pub fn ratio_to_cents(ratio: f64) -> f64 {
    1200.0 * ratio.log2()
}

// =============================================================================
// SPECTRAL HELPER FUNCTIONS
// =============================================================================

/// Generates the positive-frequency bin centres for a real-valued FFT.
///
/// This function produces the frequency coordinates corresponding to the non-redundant
/// bins of a real-input FFT of length `n_fft`. The returned vector starts at the DC
/// component (`0 Hz`) and increases monotonically up to the Nyquist frequency.
///
/// It is intended for indexing and interpreting frequency-domain outputs, plotting
/// spectra, and aligning spectral features with physical frequency units.
///
/// # Arguments
///
/// * `n_fft`\
///   The FFT size in samples. Must be greater than zero to produce meaningful results.
///
/// * `sample_rate`\
///   The sampling rate in Hertz. Must be strictly positive.
///
/// # Returns
///
/// A vector of length `n_fft / 2 + 1` containing frequency bin centres in Hertz, ordered
/// from lowest to highest frequency.
///
/// Element k is k·`sample_rate`/`n_fft` (the bin spacing of an `n_fft`-point DFT). The first
/// element is always `0 Hz`. The final element is the Nyquist frequency `sample_rate / 2`
/// for even `n_fft`, and (`n_fft` − 1)/2 · `sample_rate`/`n_fft` (below Nyquist) for odd
/// `n_fft`.
///
/// # Behavioural Guarantees
///
/// * The returned sequence is monotonically non-decreasing.
/// * The length of the returned vector is exactly `n_fft / 2 + 1`.
///
/// # Examples
///
/// ```rust
/// use audio_samples::utils::audio_math::fft_frequencies;
///
/// let freqs = fft_frequencies(1_024, 44_100.0);
///
/// assert_eq!(freqs.len(), 513);
/// assert_eq!(freqs[0], 0.0);
/// assert!(freqs.last().unwrap() <= &22_050.0);
/// ```
#[inline]
#[must_use]
pub fn fft_frequencies(n_fft: usize, sample_rate: f64) -> Vec<f64> {
    let n_bins = n_fft / 2 + 1;
    let freq_resolution = sample_rate / n_fft as f64;

    (0..n_bins).map(|i| i as f64 * freq_resolution).collect()
}

/// Generates frequencies whose spacing is uniform in the mel perceptual domain.
///
/// **This uses the HTK mel scale** (`2595 * log10(1 + f / 700)`). The Slaney scale gives
/// very different values (1 kHz is ~1000 HTK mels but 15 Slaney mels); use
/// [`mel_frequencies_with_scale`] with [`MelScale::Slaney`] for the Slaney scale.
///
/// This function samples the interval `[fmin, fmax]` evenly in the mel scale and converts
/// the resulting points back into Hertz. It is typically used when constructing mel-aligned
/// spectral representations or perceptual frequency grids.
///
/// The returned values are ordered from low to high frequency when `fmin <= fmax`.
///
/// # Arguments
///
/// * `n_mels`\
///   The number of frequency points to generate.\
///   If `n_mels == 0`, an empty vector is returned.\
///   If `n_mels == 1`, the returned vector contains only `fmin`.
///
/// * `fmin`\
///   The lower bound of the frequency range in Hertz. Values should be non-negative.
///
/// * `fmax`\
///   The upper bound of the frequency range in Hertz. Values should be non-negative.
///
/// # Returns
///
/// A vector of length `n_mels` containing frequencies in Hertz. The first element corresponds
/// approximately to `fmin` and the final element corresponds approximately to `fmax`,
/// subject to floating-point rounding.
///
/// # Behavioural Guarantees
///
/// * For `n_mels >= 2` and `fmin <= fmax`, the returned sequence is monotonically
///   non-decreasing.
/// * The endpoints are stable under round-trip conversion through the mel scale within
///   normal floating-point precision limits.
/// * The function performs no allocation beyond the returned vector and has no observable
///   side effects.
///
/// # Examples
///
/// ```rust
/// use audio_samples::utils::audio_math::mel_frequencies;
///
/// let freqs = mel_frequencies(10, 0.0, 8_000.0);
///
/// assert_eq!(freqs.len(), 10);
/// assert!((freqs[0] - 0.0).abs() < 1e-6);
/// assert!((freqs[9] - 8_000.0).abs() < 1e-6);
/// ```
#[inline]
pub fn mel_frequencies(n_mels: usize, fmin: f64, fmax: f64) -> Vec<f64> {
    let mel_min = hz_to_mel(fmin);
    let mel_max = hz_to_mel(fmax);

    let mel_points = linspace(mel_min, mel_max, n_mels);

    mel_points.into_iter().map(mel_to_hz).collect()
}

// =============================================================================
// MEL SCALES (HTK AND SLANEY)
// =============================================================================

/// The mel-scale formula used to map between Hertz and mels.
///
/// Two mel scales are in common use and they disagree substantially (by up to
/// hundreds of Hz for the same mel value), so every mel-aware function that
/// takes a `MelScale` makes the choice explicit.
///
/// - [`MelScale::Slaney`] is the scale of Malcolm Slaney's Auditory Toolbox: linear
///   (`f / (200/3)`) below 1 kHz and logarithmic (step `ln(6.4) / 27` per mel)
///   above it. It is the default.
/// - [`MelScale::Htk`] is the HTK formula `2595 * log10(1 + f / 700)`. The older
///   unparameterised helpers [`hz_to_mel`], [`mel_to_hz`],
///   [`mel_scale`] and [`mel_frequencies`] always use this scale.
///
/// # Examples
///
/// ```rust
/// use audio_samples::utils::audio_math::{MelScale, hz_to_mel_with_scale};
///
/// // 1 kHz is mel 15 on the Slaney scale and ~1000 mel on the HTK scale.
/// assert!((hz_to_mel_with_scale(1000.0, MelScale::Slaney) - 15.0).abs() < 1e-12);
/// assert!((hz_to_mel_with_scale(1000.0, MelScale::Htk) - 999.985).abs() < 1e-3);
/// assert_eq!(MelScale::default(), MelScale::Slaney);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum MelScale {
    /// Slaney (Auditory Toolbox) scale: linear below 1 kHz, logarithmic above.
    /// The default.
    #[default]
    Slaney,
    /// HTK scale: `2595 * log10(1 + f / 700)`.
    Htk,
}

/// Width of one mel on the linear part of the Slaney scale, in Hz (`200 / 3`).
const SLANEY_F_SP: f64 = 200.0 / 3.0;
/// Frequency where the Slaney scale switches from linear to logarithmic, in Hz.
const SLANEY_MIN_LOG_HZ: f64 = 1000.0;

/// Mel value at [`SLANEY_MIN_LOG_HZ`] (exactly 15).
#[inline]
fn slaney_min_log_mel() -> f64 {
    SLANEY_MIN_LOG_HZ / SLANEY_F_SP
}

/// Log-region step of the Slaney scale, `ln(6.4) / 27` (natural-log units per mel).
#[inline]
fn slaney_logstep() -> f64 {
    6.4f64.ln() / 27.0
}

/// Converts a frequency in Hertz to mels on the chosen mel scale.
///
/// This is the scale-aware counterpart of [`hz_to_mel`] (which is HTK only).
///
/// # Arguments
///
/// * `freq_hz` - Frequency in Hz. Non-negative values are meaningful; negative
///   values follow the linear branch of the Slaney formula (or produce `NaN`
///   below -700 Hz on the HTK scale).
/// * `scale` - The mel scale to use ([`MelScale::Slaney`] is the default).
///
/// # Returns
///
/// The mel value. The Slaney scale is linear (`f / (200/3)`) below 1 kHz and
/// logarithmic above (`15 + ln(f / 1000) / (ln(6.4) / 27)`); the HTK scale is
/// `2595 * log10(1 + f / 700)`.
///
/// # Behavioural Guarantees
///
/// * Monotonically increasing for `freq_hz >= 0` on both scales.
/// * [`mel_to_hz_with_scale`] with the same `scale` inverts it to within floating-point rounding.
///
/// # Examples
///
/// ```rust
/// use audio_samples::utils::audio_math::{MelScale, hz_to_mel_with_scale, mel_to_hz_with_scale};
///
/// let m = hz_to_mel_with_scale(440.0, MelScale::Slaney);
/// assert!((m - 6.6).abs() < 1e-12); // linear region: 440 / (200 / 3)
/// let back = mel_to_hz_with_scale(m, MelScale::Slaney);
/// assert!((back - 440.0).abs() < 1e-9);
/// ```
///
/// # References
///
/// M. Slaney, *Auditory Toolbox*, Technical Report 1998-010, Interval Research Corporation, 1998;
/// S. Young et al., *The HTK Book*.
#[inline]
#[must_use]
pub fn hz_to_mel_with_scale(freq_hz: f64, scale: MelScale) -> f64 {
    match scale {
        MelScale::Htk => 2595.0 * (1.0 + freq_hz / 700.0).log10(),
        MelScale::Slaney => {
            if freq_hz >= SLANEY_MIN_LOG_HZ {
                slaney_min_log_mel() + (freq_hz / SLANEY_MIN_LOG_HZ).ln() / slaney_logstep()
            } else {
                freq_hz / SLANEY_F_SP
            }
        }
    }
}

/// Converts a mel value back to Hertz on the chosen mel scale.
///
/// The inverse of [`hz_to_mel_with_scale`].
///
/// # Arguments
///
/// * `mel` - Mel value. Non-negative values are meaningful.
/// * `scale` - The mel scale to use ([`MelScale::Slaney`] is the default).
///
/// # Returns
///
/// The frequency in Hz: `(200/3) * mel` below mel 15 and
/// `1000 * exp((ln(6.4) / 27) * (mel - 15))` above it on the Slaney scale;
/// `700 * (10^(mel / 2595) - 1)` on the HTK scale.
///
/// # Examples
///
/// ```rust
/// use audio_samples::utils::audio_math::{MelScale, mel_to_hz_with_scale};
///
/// assert!((mel_to_hz_with_scale(15.0, MelScale::Slaney) - 1000.0).abs() < 1e-9);
/// assert!(mel_to_hz_with_scale(0.0, MelScale::Htk).abs() < 1e-12);
/// ```
#[inline]
#[must_use]
pub fn mel_to_hz_with_scale(mel: f64, scale: MelScale) -> f64 {
    match scale {
        MelScale::Htk => 700.0 * (10.0f64.powf(mel / 2595.0) - 1.0),
        MelScale::Slaney => {
            let min_log_mel = slaney_min_log_mel();
            if mel >= min_log_mel {
                SLANEY_MIN_LOG_HZ * (slaney_logstep() * (mel - min_log_mel)).exp()
            } else {
                SLANEY_F_SP * mel
            }
        }
    }
}

/// Generates `n_mels` frequencies evenly spaced on the chosen mel scale.
///
/// The scale-aware counterpart of [`mel_frequencies`] (HTK only). With
/// `n_mels + 2` points these are the band edges / centres used by a mel
/// filterbank built on the same scale.
///
/// # Arguments
///
/// * `n_mels` - Number of points. `0` gives an empty vector, `1` gives `[fmin]`.
/// * `fmin` - Lowest frequency in Hz.
/// * `fmax` - Highest frequency in Hz.
/// * `scale` - The mel scale to use ([`MelScale::Slaney`] is the default).
///
/// # Returns
///
/// A vector of `n_mels` frequencies in Hz, from `fmin` to `fmax` (both included,
/// up to round-off), uniformly spaced in mels.
///
/// # Examples
///
/// ```rust
/// use audio_samples::utils::audio_math::{MelScale, mel_frequencies_with_scale};
///
/// let f = mel_frequencies_with_scale(128, 0.0, 11025.0, MelScale::Slaney);
/// assert_eq!(f.len(), 128);
/// assert_eq!(f[0], 0.0);
/// assert!((f[127] - 11025.0).abs() < 1e-8);
/// ```
#[inline]
#[must_use]
pub fn mel_frequencies_with_scale(
    n_mels: usize,
    fmin: f64,
    fmax: f64,
    scale: MelScale,
) -> Vec<f64> {
    let min_mel = hz_to_mel_with_scale(fmin, scale);
    let max_mel = hz_to_mel_with_scale(fmax, scale);
    numpy_linspace(min_mel, max_mel, n_mels)
        .into_iter()
        .map(|m| mel_to_hz_with_scale(m, scale))
        .collect()
}

/// `num` evenly spaced values from `start` to `stop` inclusive: element `i` is
/// `i * step + start` (no fused multiply-add) and the last element is `stop`.
pub(crate) fn numpy_linspace(start: f64, stop: f64, num: usize) -> Vec<f64> {
    match num {
        0 => Vec::new(),
        1 => vec![start],
        _ => {
            let step = (stop - start) / (num - 1) as f64;
            let mut out: Vec<f64> = (0..num).map(|i| i as f64 * step + start).collect();
            out[num - 1] = stop;
            out
        }
    }
}

/// Generates a linearly spaced sequence of values between two endpoints.
///
/// The returned sequence includes both `start` and `end` when `num >= 2`. Values are ordered
/// monotonically when `start <= end`.
///
/// # Arguments
///
/// * `start`\
///   The starting value of the sequence.
///
/// * `end`\
///   The final value of the sequence.
///
/// * `num`\
///   The number of points to generate.\
///   If `num == 0`, an empty vector is returned.\
///   If `num == 1`, the returned vector contains only `start`.
///
/// # Returns
///
/// A vector of length `num` containing linearly interpolated values between `start` and
/// `end`.
#[inline]
#[must_use]
pub fn linspace(start: f64, end: f64, num: usize) -> Vec<f64> {
    if num == 0 {
        return Vec::new();
    }
    if num == 1 {
        return vec![start];
    }

    let step = (end - start) / (num - 1) as f64;
    (0..num).map(|i| (i as f64).mul_add(step, start)).collect()
}

// =============================================================================
// TUNING, OCTAVES, CONSTANT-Q FREQUENCIES AND KEY SPELLING
// =============================================================================

/// Reference frequency of A4 for a tuning offset, `440 * 2^(tuning / bins_per_octave)`.
#[inline]
fn tuned_a440(tuning: f64, bins_per_octave: NonZeroUsize) -> f64 {
    440.0 * 2.0f64.powf(tuning / bins_per_octave.get() as f64)
}

/// Converts a frequency in Hz to (fractional) octaves above A0.
///
/// Octave 0 is placed at `A4 / 16` (A0 = 27.5 Hz for standard tuning), so
/// 440 Hz is octave 4 and every doubling of the frequency adds 1. A non-zero
/// `tuning` moves the reference: the A4 frequency becomes
/// `440 * 2^(tuning / bins_per_octave)`.
///
/// It is used by the chroma filterbank and by tuning estimation.
///
/// # Arguments
///
/// * `freq_hz` - Frequency in Hz. Must be positive for a finite result: 0 Hz
///   gives `-inf` and negative values give `NaN`.
/// * `tuning` - Tuning deviation from A440 in fractions of a bin (for example
///   `0.25` is a quarter of a semitone sharp when `bins_per_octave` is 12).
/// * `bins_per_octave` - Number of bins per octave that `tuning` is measured in.
///
/// # Returns
///
/// `log2(freq_hz / (A4 / 16))` with `A4 = 440 * 2^(tuning / bins_per_octave)`.
///
/// # Examples
///
/// ```rust
/// use audio_samples::nzu;
/// use audio_samples::utils::audio_math::{hz_to_octs, octs_to_hz};
///
/// assert_eq!(hz_to_octs(440.0, 0.0, nzu!(12)), 4.0);
/// assert_eq!(hz_to_octs(27.5, 0.0, nzu!(12)), 0.0);
/// // A tuning of +1 bin moves the reference up one semitone.
/// assert!((hz_to_octs(440.0 * 2f64.powf(1.0 / 12.0), 1.0, nzu!(12)) - 4.0).abs() < 1e-12);
/// assert!((octs_to_hz(hz_to_octs(1000.0, 0.3, nzu!(12)), 0.3, nzu!(12)) - 1000.0).abs() < 1e-9);
/// ```
#[inline]
#[must_use]
pub fn hz_to_octs(freq_hz: f64, tuning: f64, bins_per_octave: NonZeroUsize) -> f64 {
    (freq_hz / (tuned_a440(tuning, bins_per_octave) / 16.0)).log2()
}

/// Converts (fractional) octave numbers to frequencies in Hz; the inverse of [`hz_to_octs`].
///
/// # Arguments
///
/// * `octs` - Octave number (octave 0 is `A4 / 16`, 27.5 Hz in standard tuning).
/// * `tuning` - Tuning deviation from A440 in fractions of a bin.
/// * `bins_per_octave` - Number of bins per octave that `tuning` is measured in.
///
/// # Returns
///
/// `(A4 / 16) * 2^octs` with `A4 = 440 * 2^(tuning / bins_per_octave)`.
///
/// # Examples
///
/// ```rust
/// use audio_samples::nzu;
/// use audio_samples::utils::audio_math::octs_to_hz;
///
/// assert_eq!(octs_to_hz(4.0, 0.0, nzu!(12)), 440.0);
/// assert_eq!(octs_to_hz(0.0, 0.0, nzu!(12)), 27.5);
/// ```
#[inline]
#[must_use]
pub fn octs_to_hz(octs: f64, tuning: f64, bins_per_octave: NonZeroUsize) -> f64 {
    (tuned_a440(tuning, bins_per_octave) / 16.0) * 2.0f64.powf(octs)
}

/// Converts an A4 reference frequency to a tuning deviation in fractional bins.
///
/// Computes `bins_per_octave * (log2(a4_hz) - log2(440))`. Use it to turn a concert pitch
/// such as 432 Hz or 443 Hz into the `tuning` argument that chroma, constant-Q
/// and octave helpers expect.
///
/// # Arguments
///
/// * `a4_hz` - Frequency of A4 in Hz. Must be positive for a finite result.
/// * `bins_per_octave` - Number of bins per octave to express the deviation in.
///
/// # Returns
///
/// The tuning deviation from A440 in fractions of a bin (0 for 440 Hz).
///
/// # Examples
///
/// ```rust
/// use audio_samples::nzu;
/// use audio_samples::utils::audio_math::{a4_to_tuning, tuning_to_a4};
///
/// assert_eq!(a4_to_tuning(440.0, nzu!(12)), 0.0);
/// assert!((a4_to_tuning(880.0, nzu!(12)) - 12.0).abs() < 1e-12);
/// let t = a4_to_tuning(432.0, nzu!(12)); // about -0.32 semitones
/// assert!((tuning_to_a4(t, nzu!(12)) - 432.0).abs() < 1e-9);
/// ```
#[inline]
#[must_use]
pub fn a4_to_tuning(a4_hz: f64, bins_per_octave: NonZeroUsize) -> f64 {
    bins_per_octave.get() as f64 * (a4_hz.log2() - 440.0f64.log2())
}

/// Applies [`hz_to_midi`] to every element of `freqs`.
///
/// The logarithms are evaluated with a vectorised kernel within 1 ulp of
/// the exact value (the scalar function calls the C library's `log2`), so a
/// result can differ from the scalar function's in the last bits.
///
/// # Examples
///
/// ```rust
/// use audio_samples::utils::audio_math::{hz_to_midi, hz_to_midi_slice};
///
/// let f = [27.5, 440.0, 1000.0];
/// let m = hz_to_midi_slice(&f);
/// for (a, b) in m.iter().zip(f.map(hz_to_midi)) {
///     assert!((a - b).abs() <= 1e-13 * b.abs());
/// }
/// ```
#[must_use]
pub fn hz_to_midi_slice(freqs: &[f64]) -> Vec<f64> {
    let mut out: Vec<f64> = freqs.iter().map(|&f| f / 440.0).collect();
    crate::utils::vmath::log2_f64(&mut out);
    mul_add_in_place(&mut out, 12.0, 69.0);
    out
}

/// `v = a.mul_add(v, b)` for every element. `mul_add` is a single correctly
/// rounded operation, so the hardware instruction (used when the CPU has
/// it) gives the same results as the portable call.
fn mul_add_in_place(v: &mut [f64], a: f64, b: f64) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            // SAFETY: AVX2 and FMA are available (checked above).
            unsafe { mul_add_in_place_fma(v, a, b) };
            return;
        }
    }
    v.iter_mut().for_each(|x| *x = a.mul_add(*x, b));
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
fn mul_add_in_place_fma(v: &mut [f64], a: f64, b: f64) {
    v.iter_mut().for_each(|x| *x = a.mul_add(*x, b));
}

/// Applies [`hz_to_octs`] to every element of `freqs`.
///
/// The logarithms are evaluated with a vectorised kernel within 1 ulp of
/// the exact value (the scalar function calls the C library's `log2`), so a
/// result can differ from the scalar function's in the last bits.
///
/// # Examples
///
/// ```rust
/// use audio_samples::nzu;
/// use audio_samples::utils::audio_math::{hz_to_octs, hz_to_octs_slice};
///
/// let f = [27.5, 440.0];
/// let o = hz_to_octs_slice(&f, 0.0, nzu!(12));
/// for (a, b) in o.iter().zip(f.map(|v| hz_to_octs(v, 0.0, nzu!(12)))) {
///     assert!((a - b).abs() <= 1e-13 * b.abs());
/// }
/// ```
#[must_use]
pub fn hz_to_octs_slice(freqs: &[f64], tuning: f64, bins_per_octave: NonZeroUsize) -> Vec<f64> {
    let base = tuned_a440(tuning, bins_per_octave) / 16.0;
    let mut out: Vec<f64> = freqs.iter().map(|&f| f / base).collect();
    crate::utils::vmath::log2_f64(&mut out);
    out
}

/// Applies [`a4_to_tuning`] to every element of `a4_hz`.
///
/// The logarithms are evaluated with a vectorised kernel within 1 ulp of
/// the exact value (the scalar function calls the C library's `log2`), so a
/// result can differ from the scalar function's in the last bits.
///
/// # Examples
///
/// ```rust
/// use audio_samples::nzu;
/// use audio_samples::utils::audio_math::{a4_to_tuning, a4_to_tuning_slice};
///
/// let a = [432.0, 440.0, 446.0];
/// let t = a4_to_tuning_slice(&a, nzu!(12));
/// for (x, y) in t.iter().zip(a.map(|v| a4_to_tuning(v, nzu!(12)))) {
///     assert!((x - y).abs() <= 1e-12);
/// }
/// ```
#[must_use]
pub fn a4_to_tuning_slice(a4_hz: &[f64], bins_per_octave: NonZeroUsize) -> Vec<f64> {
    let bpo = bins_per_octave.get() as f64;
    let ref_log = 440.0f64.log2();
    let mut out = a4_hz.to_vec();
    crate::utils::vmath::log2_f64(&mut out);
    out.iter_mut().for_each(|v| *v = bpo * (*v - ref_log));
    out
}

/// Converts a tuning deviation in fractional bins to the A4 reference frequency.
///
/// Computes `440 * 2^(tuning / bins_per_octave)`; the inverse of [`a4_to_tuning`].
///
/// # Arguments
///
/// * `tuning` - Tuning deviation from A440 in fractions of a bin.
/// * `bins_per_octave` - Number of bins per octave that `tuning` is measured in.
///
/// # Returns
///
/// The frequency of A4 in Hz.
///
/// # Examples
///
/// ```rust
/// use audio_samples::nzu;
/// use audio_samples::utils::audio_math::tuning_to_a4;
///
/// assert_eq!(tuning_to_a4(0.0, nzu!(12)), 440.0);
/// assert_eq!(tuning_to_a4(-12.0, nzu!(12)), 220.0);
/// ```
#[inline]
#[must_use]
pub fn tuning_to_a4(tuning: f64, bins_per_octave: NonZeroUsize) -> f64 {
    tuned_a440(tuning, bins_per_octave)
}

/// Centre frequencies of a constant-Q (geometrically spaced) filterbank.
///
/// Bin `k` has frequency `2^(tuning / bins_per_octave) * fmin * 2^(k / bins_per_octave)`.
///
/// # Arguments
///
/// * `n_bins` - Number of frequencies to generate (0 gives an empty vector).
/// * `fmin` - Frequency of bin 0 before the tuning correction, in Hz (the CQT
///   default is C1, about 32.70 Hz).
/// * `bins_per_octave` - Number of bins per octave (12 for semitones).
/// * `tuning` - Tuning deviation from A440 in fractions of a bin; all
///   frequencies are multiplied by `2^(tuning / bins_per_octave)`.
///
/// # Returns
///
/// A vector of `n_bins` increasing frequencies in Hz.
///
/// # Examples
///
/// ```rust
/// use audio_samples::nzu;
/// use audio_samples::utils::audio_math::cqt_frequencies;
///
/// let f = cqt_frequencies(25, 110.0, nzu!(12), 0.0);
/// assert_eq!(f.len(), 25);
/// assert_eq!(f[0], 110.0);
/// assert_eq!(f[12], 220.0);
/// assert_eq!(f[24], 440.0);
/// ```
#[must_use]
pub fn cqt_frequencies(
    n_bins: usize,
    fmin: f64,
    bins_per_octave: NonZeroUsize,
    tuning: f64,
) -> Vec<f64> {
    let mut out = vec![0.0; n_bins];
    cqt_frequencies_into(&mut out, fmin, bins_per_octave, tuning);
    out
}

/// [`cqt_frequencies`] written into `out` (one frequency per element, so
/// `out.len()` is the number of bins), for callers that own the output
/// buffer (e.g. a NumPy array). Same values as [`cqt_frequencies`].
///
/// # Examples
///
/// ```
/// use audio_samples::nzu;
/// use audio_samples::utils::audio_math::{cqt_frequencies, cqt_frequencies_into};
///
/// let mut out = [0.0; 24];
/// cqt_frequencies_into(&mut out, 32.70319566257483, nzu!(12), 0.0);
/// assert_eq!(out.to_vec(), cqt_frequencies(24, 32.70319566257483, nzu!(12), 0.0));
/// ```
pub fn cqt_frequencies_into(
    out: &mut [f64],
    fmin: f64,
    bins_per_octave: NonZeroUsize,
    tuning: f64,
) {
    let bpo = bins_per_octave.get() as f64;
    let correction = 2.0f64.powf(tuning / bpo);
    // Evaluation order: (correction * fmin) * 2^(k / bpo).
    for (k, o) in out.iter_mut().enumerate() {
        *o = correction * fmin * 2.0f64.powf(k as f64 / bpo);
    }
}

/// Pitch class (0 = C) of each natural note letter.
fn letter_pitch_class(letter: char) -> Option<i32> {
    match letter.to_ascii_uppercase() {
        'C' => Some(0),
        'D' => Some(2),
        'E' => Some(4),
        'F' => Some(5),
        'G' => Some(7),
        'A' => Some(9),
        'B' => Some(11),
        _ => None,
    }
}

/// Semitone offset of one accidental symbol.
fn accidental_offset(c: char) -> Option<i32> {
    match c {
        '#' | '\u{266F}' => Some(1),
        'b' | '!' | '\u{266D}' => Some(-1),
        '\u{1D12A}' => Some(2),
        '\u{1D12B}' => Some(-2),
        'n' | '\u{266E}' => Some(0),
        _ => None,
    }
}

/// Scale or mode of a parsed key signature.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum KeyKind {
    Major,
    Minor,
    /// Church mode, as the index of its 3-letter code (ion=0 .. loc=6).
    Mode(usize),
}

/// A key signature parsed by [`parse_key`].
struct ParsedKey {
    /// Upper-case tonic letter.
    letter: char,
    /// The accidental characters exactly as written.
    accidentals: String,
    kind: KeyKind,
}

impl ParsedKey {
    fn offset(&self) -> i32 {
        self.accidentals
            .chars()
            .map(|c| accidental_offset(c).unwrap_or(0))
            .sum()
    }
}

/// Parses `TONIC[accidentals]:scale-or-mode`.
fn parse_key(key: &str) -> AudioSampleResult<ParsedKey> {
    let bad = || {
        AudioSampleError::Parameter(ParameterError::invalid_value(
            "key",
            format!(
                "improper key format {key:?}: expected TONIC[accidentals]:scale, \
                 e.g. 'C:maj', 'A\u{266D}:min' or 'D:dorian'"
            ),
        ))
    };
    let mut chars = key.chars();
    let letter = chars.next().ok_or_else(bad)?;
    if letter_pitch_class(letter).is_none() || !letter.is_ascii_alphabetic() {
        return Err(bad());
    }
    let rest: String = chars.collect();
    let (accidentals, kind) = rest.split_once(':').ok_or_else(bad)?;
    if !accidentals.chars().all(|c| accidental_offset(c).is_some()) {
        return Err(bad());
    }
    let kind = match kind {
        "maj" | "major" => KeyKind::Major,
        "min" | "minor" => KeyKind::Minor,
        "ion" | "ionian" => KeyKind::Mode(0),
        "dor" | "dorian" => KeyKind::Mode(1),
        "phr" | "phryg" | "phrygian" => KeyKind::Mode(2),
        "lyd" | "lydian" => KeyKind::Mode(3),
        "mix" | "mixolyd" | "mixolydian" => KeyKind::Mode(4),
        "aeo" | "aeol" | "aeolian" => KeyKind::Mode(5),
        "loc" | "locr" | "locrian" => KeyKind::Mode(6),
        _ => return Err(bad()),
    };
    Ok(ParsedKey {
        letter: letter.to_ascii_uppercase(),
        accidentals: accidentals.to_string(),
        kind,
    })
}

/// Pitch class of a note name made of a letter and accidental symbols.
fn note_name_degree(note: &str) -> i32 {
    let mut chars = note.chars();
    let letter = chars.next().and_then(letter_pitch_class).unwrap_or(0);
    let acc: i32 = chars.map(|c| accidental_offset(c).unwrap_or(0)).sum();
    (letter + acc).rem_euclid(12)
}

/// Cancels accidentals and spells the net offset with
/// (double) sharps or flats, Unicode symbols.
fn simplify_note(note: &str) -> String {
    let mut chars = note.chars();
    let letter = chars.next().unwrap_or('C').to_ascii_uppercase();
    let offset: i32 = chars.map(|c| accidental_offset(c).unwrap_or(0)).sum();
    let mut out = String::new();
    out.push(letter);
    let (single, double) = if offset >= 0 {
        ('\u{266F}', '\u{1D12A}')
    } else {
        ('\u{266D}', '\u{1D12B}')
    };
    if offset.rem_euclid(2) == 1 {
        out.push(single);
    }
    for _ in 0..offset.abs() / 2 {
        out.push(double);
    }
    out
}

/// Maps Unicode accidentals to their ASCII forms.
fn ascii_accidentals(note: &str) -> String {
    let mut out = String::with_capacity(note.len());
    for c in note.chars() {
        match c {
            '\u{266F}' => out.push('#'),
            '\u{1D12A}' => out.push_str("##"),
            '\u{266D}' => out.push('b'),
            '\u{1D12B}' => out.push_str("bb"),
            '\u{266E}' => out.push('n'),
            other => out.push(other),
        }
    }
    out
}

/// The relative major key of a mode (Unicode spelling).
fn mode_to_major(letter: char, accidentals: &str, mode: usize) -> String {
    // MAJOR_DICT[mode][letter], indexed by mode (ion..loc) and letter (C D E F G A B).
    const MAJOR: [[&str; 7]; 7] = [
        ["C", "D", "E", "F", "G", "A", "B"],
        ["B\u{266D}", "C", "D", "E\u{266D}", "F", "G", "A"],
        [
            "A\u{266D}",
            "B\u{266D}",
            "C",
            "D\u{266D}",
            "E\u{266D}",
            "F",
            "G",
        ],
        ["G", "A", "B", "C", "D", "E", "F\u{266F}"],
        ["F", "G", "A", "B\u{266D}", "C", "D", "E"],
        ["E\u{266D}", "F", "G", "A\u{266D}", "B\u{266D}", "C", "D"],
        [
            "D\u{266D}",
            "E\u{266D}",
            "F",
            "G\u{266D}",
            "A\u{266D}",
            "B\u{266D}",
            "C",
        ],
    ];
    let col = "CDEFGAB".find(letter).unwrap_or(0);
    let tonic = format!("{}{accidentals}", MAJOR[mode][col]);
    format!("{}:maj", simplify_note(&tonic))
}

/// Lists the 12 chromatic note names (C first) as spelled in a given key or mode.
///
/// It resolves enharmonic spellings (C♯ vs D♭) for human-readable output such as
/// plot labels: a tonic with an accidental fixes sharps or flats, otherwise the
/// spelling minimises accidentals on the scale degrees (sharps win ties);
/// keys with six or more sharps or flats use E♯, B♯, C♭, F♭ and double
/// accidentals where needed.
///
/// # Arguments
///
/// * `key` - Key signature `TONIC[accidentals]:scale`. The tonic is a letter
///   `A`-`G` (either case); accidentals are any of `#` `♯` `𝄪` `b` `!` `♭` `𝄫`
///   `♮` `n` (combinations allowed); the scale is `maj`, `major`, `min`,
///   `minor`, or a mode: `ionian`, `dorian`, `phrygian`, `lydian`,
///   `mixolydian`, `aeolian`, `locrian`, abbreviated to its first three
///   letters or without `ian` (e.g. `mix`, `mixolyd`). Scale and mode names
///   are lower case.
/// * `unicode` - If `true`, accidentals are the Unicode symbols ♯ 𝄪 ♭ 𝄫 ♮;
///   if `false` they are mapped to `#`, `##`, `b`, `bb`, `n`.
/// * `natural` - If `true`, notes outside the scale that carry no accidental
///   get a natural sign (`D♮` in `G:loc`).
///
/// # Returns
///
/// Twelve note names; element `k` names pitch class `k` (0 = C).
///
/// # Errors
///
/// [`ParameterError`] if `key` does not follow the format above.
///
/// # Examples
///
/// ```rust
/// use audio_samples::utils::audio_math::key_to_notes;
///
/// let c = key_to_notes("C:maj", false, false).unwrap();
/// assert_eq!(c, ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]);
/// let eb = key_to_notes("Eb:min", true, false).unwrap();
/// assert_eq!(eb[11], "C\u{266D}");
/// assert_eq!(key_to_notes("G#:maj", false, false).unwrap()[7], "F##");
/// assert!(key_to_notes("H:maj", true, false).is_err());
/// ```
pub fn key_to_notes(key: &str, unicode: bool, natural: bool) -> AudioSampleResult<Vec<String>> {
    let parsed = parse_key(key)?;
    let scale_is_major = match parsed.kind {
        KeyKind::Mode(mode) => {
            let equiv = mode_to_major(parsed.letter, &parsed.accidentals, mode);
            return key_to_notes(&equiv, unicode, natural);
        }
        KeyKind::Major => true,
        KeyKind::Minor => false,
    };
    let offset = parsed.offset();
    let tonic = parsed.letter;

    if offset.abs() >= 2 {
        // Multiple accidentals: spell the key one accidental closer to natural,
        // add the extra accidental to every note, then rotate C's spelling to the front.
        let acc = if offset > 0 { '\u{266F}' } else { '\u{266D}' };
        let reduced: String =
            std::iter::repeat_n(acc, (offset.unsigned_abs() - 1) as usize).collect();
        let scale = if scale_is_major { "maj" } else { "min" };
        let intermediate = key_to_notes(&format!("{tonic}{reduced}:{scale}"), true, false)?;
        let mut notes: Vec<String> = intermediate
            .iter()
            .map(|n| simplify_note(&format!("{n}{acc}")))
            .collect();
        let zero = notes
            .iter()
            .position(|n| note_name_degree(n) == 0)
            .unwrap_or(0);
        notes.rotate_left(zero);
        if !unicode {
            notes = notes.iter().map(|n| ascii_accidentals(n)).collect();
        }
        return Ok(notes);
    }

    let pc = letter_pitch_class(tonic).unwrap_or(0);
    // Steps clockwise on the circle of fifths (= number of sharps).
    let tonic_number = if scale_is_major {
        ((pc + offset) * 7).rem_euclid(12)
    } else {
        ((pc + offset) * 7 + 9).rem_euclid(12)
    };
    let use_sharps = match offset {
        o if o < 0 => false,
        o if o > 0 => true,
        // A natural tonic never lands on 6 (that needs F♯ / E♭), so `<= 6` is total.
        _ => tonic_number <= 6,
    };

    let mut notes: Vec<String> = if use_sharps {
        let mut n: Vec<String> = [
            "C",
            "C\u{266F}",
            "D",
            "D\u{266F}",
            "E",
            "F",
            "F\u{266F}",
            "G",
            "G\u{266F}",
            "A",
            "A\u{266F}",
            "B",
        ]
        .iter()
        .map(|s| (*s).to_string())
        .collect();
        const SHARP_CORRECTIONS: [(usize, &str); 7] = [
            (5, "E\u{266F}"),
            (0, "B\u{266F}"),
            (7, "F\u{1D12A}"),
            (2, "C\u{1D12A}"),
            (9, "G\u{1D12A}"),
            (4, "D\u{1D12A}"),
            (11, "A\u{1D12A}"),
        ];
        // Distinguish B♯:maj (12 sharps) from C:maj (0).
        let n_sharps = if tonic_number == 0 && tonic == 'B' {
            12
        } else {
            tonic_number
        };
        for &(index, name) in SHARP_CORRECTIONS
            .iter()
            .take((n_sharps - 5).max(0) as usize)
        {
            n[index] = name.to_string();
        }
        n
    } else {
        let mut n: Vec<String> = [
            "C",
            "D\u{266D}",
            "D",
            "E\u{266D}",
            "E",
            "F",
            "G\u{266D}",
            "G",
            "A\u{266D}",
            "A",
            "B\u{266D}",
            "B",
        ]
        .iter()
        .map(|s| (*s).to_string())
        .collect();
        const FLAT_CORRECTIONS: [(usize, &str); 6] = [
            (11, "C\u{266D}"),
            (4, "F\u{266D}"),
            (9, "B\u{1D12B}"),
            (2, "E\u{1D12B}"),
            (7, "A\u{1D12B}"),
            (0, "D\u{1D12B}"),
        ];
        let n_flats = (12 - tonic_number).rem_euclid(12);
        for &(index, name) in FLAT_CORRECTIONS.iter().take((n_flats - 5).max(0) as usize) {
            n[index] = name.to_string();
        }
        n
    };

    if natural {
        let degrees = key_to_degrees(key)?;
        for note in &mut notes {
            let d = note_name_degree(note);
            if degrees.iter().any(|&s| i32::from(s) == d) {
                continue;
            }
            if note.chars().count() == 1 {
                note.push('\u{266E}');
            }
        }
    }
    if !unicode {
        notes = notes.iter().map(|n| ascii_accidentals(n)).collect();
    }
    Ok(notes)
}

/// [`key_to_notes`] as a reference to a cached table: the first call for a
/// key (and flags) computes the twelve names and keeps them for the life of
/// the process (one entry per distinct key string); later calls allocate
/// nothing.
///
/// # Errors
///
/// As [`key_to_notes`].
///
/// # Examples
///
/// ```rust
/// use audio_samples::utils::audio_math::{key_to_notes, key_to_notes_static};
///
/// let notes = key_to_notes_static("F#:maj", true, false).unwrap();
/// assert_eq!(notes.to_vec(), key_to_notes("F#:maj", true, false).unwrap());
/// ```
pub fn key_to_notes_static(
    key: &str,
    unicode: bool,
    natural: bool,
) -> AudioSampleResult<&'static [&'static str; 12]> {
    type Entry = [Option<&'static [&'static str; 12]>; 4];
    static CACHE: std::sync::LazyLock<std::sync::Mutex<std::collections::HashMap<String, Entry>>> =
        std::sync::LazyLock::new(Default::default);
    let slot = usize::from(unicode) * 2 + usize::from(natural);
    if let Some(&Some(t)) = CACHE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .get(key)
        .map(|e| &e[slot])
    {
        return Ok(t);
    }
    let notes = key_to_notes(key, unicode, natural)?;
    let names: Vec<&'static str> = notes
        .into_iter()
        .map(|n| &*Box::leak(n.into_boxed_str()))
        .collect();
    let table: [&'static str; 12] = names.try_into().expect("key_to_notes returns twelve names");
    let table: &'static [&'static str; 12] = Box::leak(Box::new(table));
    let mut cache = CACHE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let entry = cache.entry(key.to_string()).or_insert([None; 4]);
    // A concurrent first call may have filled the slot; keep the first.
    Ok(*entry[slot].get_or_insert(table))
}

/// The seven diatonic scale degrees of a key or mode, as pitch classes (0 = C).
///
/// These are the pitch classes of the scale starting from the tonic. Major keys
/// use the steps 0 2 4 5 7 9 11, minor keys 0 2 3 5 7 8 10, and modes rotate the
/// relative major scale.
///
/// # Arguments
///
/// * `key` - Key signature in the format accepted by [`key_to_notes`]
///   (e.g. `"C:maj"`, `"A\u{266D}:min"`, `"D:dorian"`, `"C#b#:min"`).
///
/// # Returns
///
/// Seven pitch classes in `0..12`, tonic first.
///
/// # Errors
///
/// [`ParameterError`] if `key` is not a valid key signature.
///
/// # Examples
///
/// ```rust
/// use audio_samples::utils::audio_math::key_to_degrees;
///
/// assert_eq!(key_to_degrees("C:maj").unwrap(), [0, 2, 4, 5, 7, 9, 11]);
/// assert_eq!(key_to_degrees("C#:maj").unwrap(), [1, 3, 5, 6, 8, 10, 0]);
/// assert_eq!(key_to_degrees("A:min").unwrap(), [9, 11, 0, 2, 4, 5, 7]);
/// assert_eq!(key_to_degrees("D:dorian").unwrap(), [2, 4, 5, 7, 9, 11, 0]);
/// ```
pub fn key_to_degrees(key: &str) -> AudioSampleResult<[u8; 7]> {
    const MAJOR: [i32; 7] = [0, 2, 4, 5, 7, 9, 11];
    const MINOR: [i32; 7] = [0, 2, 3, 5, 7, 8, 10];
    let parsed = parse_key(key)?;
    let steps = match parsed.kind {
        KeyKind::Mode(mode) => {
            let equiv = mode_to_major(parsed.letter, &parsed.accidentals, mode);
            let mut degrees = key_to_degrees(&equiv)?;
            degrees.rotate_left(mode);
            return Ok(degrees);
        }
        KeyKind::Major => MAJOR,
        KeyKind::Minor => MINOR,
    };
    let base = letter_pitch_class(parsed.letter).unwrap_or(0) + parsed.offset();
    let mut out = [0u8; 7];
    for (o, s) in out.iter_mut().zip(steps) {
        // rem_euclid(12) is in 0..12, so the cast is lossless.
        *o = (s + base).rem_euclid(12) as u8;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_note_midi_conversions() {
        // Test A4
        assert_eq!(note_to_midi("A4").unwrap(), 69);
        assert_eq!(midi_to_note(69).unwrap(), "A4");

        // Test C4 (middle C)
        assert_eq!(note_to_midi("C4").unwrap(), 60);
        assert_eq!(midi_to_note(60).unwrap(), "C4");

        // Test sharp/flat equivalence
        assert_eq!(note_to_midi("C#4").unwrap(), note_to_midi("Db4").unwrap());

        // Lowercase letters are accepted.
        assert_eq!(note_to_midi("a4").unwrap(), 69);
    }

    /// The parser accepts case-insensitive letters, stacked
    /// ASCII/Unicode accidentals that may cross the octave boundary, signed
    /// octaves and an optional cents suffix (rounded half to even).
    #[test]
    fn note_to_midi_accepts_librosa_grammar() {
        let cases: &[(&str, u8)] = &[
            ("C4", 60),
            ("c4", 60),
            ("B#3", 60),
            ("Cb4", 59),
            ("E#4", 65),
            ("Fb4", 64),
            ("C##4", 62),
            ("Dbb4", 60),
            ("C\u{266F}4", 61),
            ("D\u{266D}4", 61),
            ("C\u{1D12A}4", 62),
            ("D\u{1D12B}4", 60),
            ("C\u{266E}4", 60),
            ("C!4", 59),
            ("C-1", 0),
            ("G9", 127),
            ("C+4", 60),
            ("A4+50", 70),
            ("A4-50", 68),
            ("A4+49", 69),
            ("C-1+40", 0),
        ];
        for &(name, midi) in cases {
            assert_eq!(note_to_midi(name).unwrap(), midi, "{name}");
        }
        // Every name midi_to_note produces parses back.
        for midi in 0..=127u8 {
            assert_eq!(note_to_midi(&midi_to_note(midi).unwrap()).unwrap(), midi);
        }
        // Outside 0..=127 after parsing: a range error.
        for name in ["Cb-1", "G#9", "C10", "C-2"] {
            assert!(
                matches!(
                    note_to_midi(name),
                    Err(AudioSampleError::Parameter(
                        ParameterError::OutOfRange { .. }
                    ))
                ),
                "{name}"
            );
        }
    }

    #[test]
    fn note_to_midi_rejects_malformed_with_spans() {
        // (input, span offset, span length, message fragment)
        let cases: &[(&str, usize, usize, &str)] = &[
            ("", 0, 0, "octave number"),
            ("A", 0, 1, "octave number"),
            ("Bb", 0, 2, "octave number"),
            ("H4", 0, 1, "unrecognised note"),
            ("\u{266F}4", 0, 3, "unrecognised note"),
            ("Cn4", 1, 2, "invalid octave"),
            ("C4x", 2, 1, "invalid cents"),
            ("A4+", 2, 1, "invalid cents"),
            ("A4+5x", 2, 3, "invalid cents"),
        ];
        for &(input, offset, len, kind) in cases {
            match note_to_midi(input) {
                Err(AudioSampleError::NoteParse(npe)) => {
                    assert_eq!(npe.span.offset(), offset, "{input:?}");
                    assert_eq!(npe.span.len(), len, "{input:?}");
                    assert!(npe.kind.contains(kind), "{input:?}: {}", npe.kind);
                }
                other => panic!("{input:?}: expected NoteParse, got {other:?}"),
            }
        }
    }

    /// Below about 7.9 Hz the nearest MIDI note is negative. The old code cast it
    /// with a saturating `as u8`, labelling e.g. 4 Hz and 1 Hz as "C-1" with cents
    /// measured against a different note. It is now a range error, like
    /// frequencies above G9.
    #[test]
    fn frequency_to_note_rejects_notes_outside_midi_range() {
        for f in [1.0, 4.0, 7.9, 13_000.0, 20_000.0] {
            assert!(
                matches!(
                    frequency_to_note(f),
                    Err(AudioSampleError::Parameter(
                        ParameterError::OutOfRange { .. }
                    ))
                ),
                "{f} Hz"
            );
        }
        for f in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(frequency_to_note(f).is_err(), "{f} Hz");
        }
        // Edges of the range still work.
        let (note, cents) = frequency_to_note(8.1758).unwrap();
        assert_eq!(note, "C-1");
        assert!(cents.abs() < 0.1);
        let (note, _) = frequency_to_note(12_543.85).unwrap();
        assert_eq!(note, "G9");
    }

    #[test]
    fn test_note_parse_error_spans() {
        use crate::AudioSampleError;
        use miette::Diagnostic;

        // Unrecognised note letter: caret points at the leading note token.
        let err = note_to_midi("H4").unwrap_err();
        match &err {
            AudioSampleError::NoteParse(npe) => {
                assert_eq!(npe.span.offset(), 0);
                assert_eq!(npe.span.len(), 1); // the "H"
                assert!(npe.kind.contains("unrecognised note"));
            }
            other => panic!("expected NoteParse, got {other:?}"),
        }
        assert_eq!(
            err.code().unwrap().to_string(),
            "audio_samples::parse::note_name"
        );
        assert!(err.help().unwrap().to_string().contains("A4"));

        // Bad octave: caret moves past the note token to the octave digits.
        let err = note_to_midi("C#x").unwrap_err();
        match err {
            AudioSampleError::NoteParse(npe) => {
                assert_eq!(npe.span.offset(), 2); // after "C#"
                assert_eq!(npe.span.len(), 1); // the "x"
                assert!(npe.kind.contains("invalid octave"));
            }
            other => panic!("expected NoteParse, got {other:?}"),
        }

        // A note that parses but lands outside the MIDI range is a genuine
        // range error, not a parse error. B9 maps to MIDI 131 (> 127).
        let err = note_to_midi("B9").unwrap_err();
        assert!(matches!(
            err,
            AudioSampleError::Parameter(crate::ParameterError::OutOfRange { .. })
        ));
    }

    #[test]
    fn test_note_frequency_conversion() {
        let freq = note_to_frequency("A4").unwrap();
        assert!(
            (freq - 440.0f64).abs() < 0.001f64,
            "Frequency of A4 should be near 440 Hz --- {}",
            freq
        );

        let (note, cents): (String, f64) = frequency_to_note(440.0f64).unwrap();
        assert_eq!(note, "A4");
        assert!(
            cents.abs() < 1.0f64,
            "Cents deviation should be near zero --- {}",
            cents
        );
    }

    #[test]
    fn test_cents_ratio_conversions() {
        // Test octave (1200 cents = ratio of 2)
        assert!((cents_to_ratio(1200.0f64) - 2.0f64).abs() < 0.001f64);
        assert!((ratio_to_cents(2.0f64) - 1200.0f64).abs() < 0.001f64);

        // Test semitone (100 cents ≈ ratio of 1.059)
        assert!((cents_to_ratio(100.0f64) - 1.059463f64).abs() < 0.001f64);
        assert!((ratio_to_cents(1.059463f64) - 100.0f64).abs() < 0.1f64);
    }

    #[test]
    fn test_time_frame_conversions() {
        let sample_rate = 44100.0;
        let hop_size = 512;

        // Test frame to time conversion
        let time = frames_to_time(100, sample_rate, hop_size);
        let expected_time = (100 * hop_size) as f64 / sample_rate;
        assert!((time - expected_time).abs() < 0.001f64);

        // Test round-trip
        let frames = time_to_frames(time, sample_rate, hop_size);
        assert_eq!(frames, 100);
    }

    #[test]
    fn test_fft_frequencies() {
        let freqs = fft_frequencies(1024, 44100.0);
        assert_eq!(freqs.len(), 513); // 1024/2 + 1
        assert_eq!(freqs[0], 0.0); // DC
        assert!((freqs[freqs.len() - 1] - 22050.0f64).abs() < 0.1f64); // Nyquist

        // Check frequency resolution
        let freq_res = freqs[1] - freqs[0];
        let expected_res = 44100.0 / 1024.0;
        assert!((freq_res - expected_res).abs() < 0.001f64);
    }

    #[test]
    fn htk_mel_round_trip_regression() {
        // Regression for the old `mel_to_hz` formula (700 * 10^(m/2595) - 1), which was off
        // by +699 Hz at every frequency.
        for &f in &[0.0, 50.0, 440.0, 1000.0, 8000.0, 22050.0] {
            let back = mel_to_hz(hz_to_mel(f));
            assert!((back - f).abs() <= 1e-9 * f.max(1.0), "{f} -> {back}");
            let back2 = mel_to_hz_with_scale(hz_to_mel_with_scale(f, MelScale::Htk), MelScale::Htk);
            assert_eq!(back, back2);
        }
        assert_eq!(mel_to_hz(0.0), 0.0);
    }

    #[test]
    fn slaney_mel_scale_values() {
        // Linear region: f / (200 / 3); break point 1 kHz = mel 15.
        assert!((hz_to_mel_with_scale(200.0, MelScale::Slaney) - 3.0).abs() < 1e-12);
        assert!((hz_to_mel_with_scale(1000.0, MelScale::Slaney) - 15.0).abs() < 1e-12);
        // Log region: 6400 Hz is exactly 27 mels above 1 kHz (logstep = ln(6.4)/27).
        assert!((hz_to_mel_with_scale(6400.0, MelScale::Slaney) - 42.0).abs() < 1e-12);
        for &f in &[0.0, 1.0, 999.0, 1000.0, 1001.0, 11025.0, 44100.0] {
            let m = hz_to_mel_with_scale(f, MelScale::Slaney);
            let back = mel_to_hz_with_scale(m, MelScale::Slaney);
            assert!((back - f).abs() <= 1e-10 * f.max(1.0), "{f} -> {back}");
        }
        // HTK and Slaney genuinely differ.
        assert!(hz_to_mel_with_scale(1000.0, MelScale::Htk) > 999.0);
    }

    #[test]
    fn mel_frequencies_with_scale_matches_htk_helper_and_endpoints() {
        let a = mel_frequencies_with_scale(40, 133.0, 6854.0, MelScale::Htk);
        let b = mel_frequencies(40, 133.0, 6854.0);
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 1e-9);
        }
        let s = mel_frequencies_with_scale(5, 0.0, 8000.0, MelScale::Slaney);
        assert_eq!(s[0], 0.0);
        assert!((s[4] - 8000.0).abs() < 1e-9);
        assert!(s.windows(2).all(|w| w[1] > w[0]));
        assert!(mel_frequencies_with_scale(0, 0.0, 1.0, MelScale::Slaney).is_empty());
        assert_eq!(
            mel_frequencies_with_scale(1, 50.0, 100.0, MelScale::Slaney).len(),
            1
        );
    }

    #[test]
    fn numpy_linspace_matches_numpy_rounding() {
        let v = numpy_linspace(0.0, 1.0, 11);
        assert_eq!(v.len(), 11);
        assert_eq!(v[3], 3.0 * 0.1); // i * step, not fused
        assert_eq!(v[10], 1.0);
        assert_eq!(numpy_linspace(2.0, 5.0, 1), vec![2.0]);
        assert!(numpy_linspace(2.0, 5.0, 0).is_empty());
    }

    #[test]
    fn librosa_frame_conversions() {
        let hop = NonZeroUsize::new(512).unwrap();
        // Floor division, including negatives.
        assert_eq!(samples_to_frames(-1, hop, None), -1);
        assert_eq!(samples_to_frames(-512, hop, None), -1);
        assert_eq!(samples_to_frames(-513, hop, None), -2);
        assert_eq!(samples_to_frames(1024, hop, Some(2048)), 0);
        assert_eq!(samples_to_frames(1023, hop, Some(2048)), -1);
        // Odd n_fft: offset is n_fft / 2 (integer halving).
        assert_eq!(frames_to_samples(0, hop, Some(2047)), 1023);
        // Saturation instead of overflow.
        assert_eq!(frames_to_samples(i64::MAX, hop, Some(4)), i64::MAX);
        // Truncation toward zero.
        assert_eq!(time_to_samples(1.9999 / 100.0, 100.0), 1);
        assert_eq!(time_to_samples(-1.9999 / 100.0, 100.0), -1);
        // Composition.
        assert_eq!(time_to_frames_floor(-0.1, 22050.0, hop, None), -5);
        assert_eq!(time_to_frames_floor(0.9, 22050.0, hop, None), 38);
        // The rounding helper differs from the floor helper here (documented).
        assert_eq!(time_to_frames(0.9, 22050.0, 512), 39);
        assert_eq!(frames_to_time_with_offset(-2, 1000.0, hop, None), -1.024);
        assert_eq!(
            times_like(2, 1000.0, NonZeroUsize::new(10).unwrap(), Some(4)),
            vec![0.002, 0.012]
        );
        assert_eq!(
            samples_like(2, NonZeroUsize::new(10).unwrap(), Some(4)),
            vec![2, 12]
        );
    }

    #[test]
    fn slice_conversions_match_scalar_functions() {
        let mut seed = 1u64;
        let mut next = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            seed
        };
        let mut floats: Vec<f64> = vec![
            0.0,
            -0.0,
            1.0,
            -1.0,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            1e300,
            -1e300,
            9.3e18,
            -9.3e18,
            4.5e15,
            f64::MIN_POSITIVE,
        ];
        for _ in 0..4000 {
            let u = (next() >> 11) as f64 / (1u64 << 53) as f64;
            floats.push(u * 30000.0);
            floats.push((u - 0.5) * 120.0);
        }
        let mut ints: Vec<i64> = vec![
            0,
            -1,
            1,
            i64::MIN,
            i64::MAX,
            (1 << 52) - 1,
            1 << 52,
            -(1 << 52),
        ];
        for _ in 0..4000 {
            ints.push((next() >> 40) as i64);
            ints.push((next() >> 12) as i64 - (1 << 51));
            ints.push(next() as i64);
        }
        for hop in [
            1usize,
            2,
            3,
            7,
            512,
            513,
            1 << 51,
            (1 << 52) + 1,
            1 << 62,
            (1 << 62) + 3,
            i64::MAX as usize,
        ] {
            let h = NonZeroUsize::new(hop).unwrap();
            for n_fft in [None, Some(2048)] {
                let want: Vec<i64> = ints
                    .iter()
                    .map(|&s| samples_to_frames(s, h, n_fft))
                    .collect();
                assert_eq!(samples_to_frames_slice(&ints, h, n_fft), want, "hop {hop}");
                for sr in [22050.0, 16000.0, 1.0] {
                    let want: Vec<i64> = floats
                        .iter()
                        .map(|&t| time_to_frames_floor(t, sr, h, n_fft))
                        .collect();
                    assert_eq!(time_to_frames_floor_slice(&floats, sr, h, n_fft), want);
                }
            }
        }
        for sr in [22050.0, 1e10, 0.5] {
            let want: Vec<i64> = floats.iter().map(|&t| time_to_samples(t, sr)).collect();
            assert_eq!(time_to_samples_slice(&floats, sr), want);
        }
        let pos: Vec<f64> = floats.iter().map(|v| v.abs()).collect();
        let bpo = NonZeroUsize::new(12).unwrap();
        // The log2 conversions use a vectorised log2 within 1 ulp of the
        // C library's.
        let close = |a: Vec<f64>, b: Vec<f64>| {
            for (x, y) in a.into_iter().zip(b) {
                assert!(
                    x.to_bits() == y.to_bits() || (x - y).abs() <= 1e-12 * y.abs().max(1.0),
                    "{x} vs {y}"
                );
            }
        };
        close(
            hz_to_midi_slice(&pos),
            pos.iter().map(|&f| hz_to_midi(f)).collect(),
        );
        close(
            hz_to_octs_slice(&pos, 0.3, bpo),
            pos.iter().map(|&f| hz_to_octs(f, 0.3, bpo)).collect(),
        );
        close(
            a4_to_tuning_slice(&pos, bpo),
            pos.iter().map(|&f| a4_to_tuning(f, bpo)).collect(),
        );
    }

    #[test]
    fn static_note_names_match_owned() {
        for m in 0..=255u8 {
            match (midi_to_note(m), midi_to_note_static(m)) {
                (Ok(a), Ok(b)) => assert_eq!(a, b),
                (Err(a), Err(b)) => assert_eq!(format!("{a:?}"), format!("{b:?}")),
                _ => panic!("midi {m}"),
            }
        }
        for f in [
            -1.0,
            0.0,
            1.0,
            7.0,
            8.2,
            27.5,
            440.0,
            445.0,
            1000.0,
            12543.0,
            13000.0,
            f64::NAN,
        ] {
            match (frequency_to_note(f), frequency_to_note_static(f)) {
                (Ok((a, ca)), Ok((b, cb))) => assert_eq!((a.as_str(), ca), (b, cb)),
                (Err(a), Err(b)) => assert_eq!(format!("{a:?}"), format!("{b:?}")),
                _ => panic!("freq {f}"),
            }
        }
        for key in [
            "C:maj", "F#:maj", "Eb:min", "C##:maj", "Bbb:min", "D:dorian", "X:maj",
        ] {
            for (u, n) in [(false, false), (true, false), (false, true), (true, true)] {
                for _ in 0..2 {
                    match (key_to_notes(key, u, n), key_to_notes_static(key, u, n)) {
                        (Ok(a), Ok(b)) => assert_eq!(a, b.to_vec(), "{key} {u} {n}"),
                        (Err(_), Err(_)) => {}
                        _ => panic!("key {key}"),
                    }
                }
            }
        }
    }
}
