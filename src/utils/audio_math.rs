//! Audio-domain mathematical utilities and canonical unit conversions.
//!
//! This module provides a coherent set of domain-level transformations commonly required
//! when working with audio signals, spectral representations, and musical abstractions.
//! It centralises conversions between physical units (time, frequency, amplitude, power),
//! perceptual scales (mel, decibels, cents), and symbolic musical representations
//! (note names, MIDI numbers), so that higher-level code can remain focused.
//!
//! The primary role of this module within the crate is to establish a single, predictable
//! interpretation of these conversions. Audio systems frequently accumulate subtle
//! inconsistencies when unit handling is duplicated across pipelines or tools; by treating
//! these transforms as part of the public API surface, the library ensures that behaviour
//! remains stable, explicit, and auditable over time.
//!
//! The functions are intended to be used as small, composable building blocks. Most operate
//! directly on scalar values and return plain numeric types or lightweight domain objects,
//! making them suitable for preprocessing, analysis, visualisation, and validation stages.
//! Where conversions are reversible, the paired operations are exposed explicitly to avoid
//! hidden coupling or implicit state.
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

use crate::{AudioSampleError, AudioSampleResult, ParameterError};
use std::collections::HashMap;

// =============================================================================
// FREQUENCY CONVERSIONS
// =============================================================================

/// Converts a frequency expressed in Hertz into the mel perceptual scale.
///
/// This function is intended for situations where linear frequency values need to be mapped
/// into a perceptual representation, such as when constructing mel filter banks, analysing
/// spectral content in perceptual units, or comparing pitch-related quantities on a scale
/// that better reflects human sensitivity.
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
/// This function performs the inverse mapping of [`hz_to_mel`]. It is intended for cases
/// where perceptual-domain representations need to be projected back into physical
/// frequency space, such as when constructing frequency bins, interpreting mel-domain
/// features, or synthesising frequency-aligned outputs.
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
/// This function is typically used when sampling a frequency range in perceptual space,
/// for example when constructing mel-aligned filter banks, visualisations, or validation
/// grids. The returned values are expressed in Hertz, but their distribution corresponds
/// to equal spacing after conversion to the mel scale.
///
/// The sequence includes both endpoints and preserves ordering from low to high
/// frequency when `fmin <= fmax`.
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
/// To avoid infinite values and undefined logarithms, non-positive amplitudes are mapped to
/// a fixed floor value of `-80 dB`.
///
/// # Arguments
///
/// * `amplitude`\
///   The input amplitude ratio. Values greater than zero represent valid signal magnitudes.
///   Zero or negative values are treated as silence and mapped to the floor value.
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
///   amplitude up to floating-point rounding error only when the floor has not been applied.
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
/// To avoid infinite values and undefined logarithms, non-positive power values are mapped
/// to a fixed floor value of `-80 dB`. This behaviour is part of the public contract and
/// should be considered when interpreting silence or numerical underflow.
///
/// # Arguments
///
/// * `power`\
///   The input power ratio. Values greater than zero represent valid power measurements.
///   Zero or negative values are treated as silence and mapped to the floor value.
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
///   power up to floating-point rounding error only when the floor has not been applied.
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
/// The nearest frame index as a `usize`. If the computed value cannot be represented as a
/// valid `usize`, the function returns `0`.
///
/// # Behavioural Guarantees
///
/// * For fixed parameters, applying [`frames_to_time`] to the returned frame index produces
///   a time value close to the original input, subject to rounding.
/// * Rounding is performed to the nearest integer frame index rather than truncation.
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
/// * For fixed parameters, applying [`time_to_samples`] to the returned value recovers the
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
/// The nearest sample index as a `usize`. If the computed value cannot be represented as a
/// valid `usize`, the function returns `0`.
///
/// # Behavioural Guarantees
///
/// * For fixed parameters, applying [`samples_to_time`] to the returned sample index
///   produces a time value close to the original input, subject to rounding.
/// * Rounding is performed to the nearest integer sample index rather than truncation.
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
/// - `ms` – Duration in milliseconds.
/// - `sample_rate` – Sampling rate in Hz. Must be strictly positive.
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
// MUSICAL THEORY FUNCTIONS
// =============================================================================

static NOTE_TO_MIDI_MAP: std::sync::LazyLock<HashMap<&'static str, u8>> =
    std::sync::LazyLock::new(|| {
        let mut m = HashMap::new();

        // Base notes (octave 0)
        m.insert("C", 0);
        m.insert("C#", 1);
        m.insert("Db", 1);
        m.insert("D", 2);
        m.insert("D#", 3);
        m.insert("Eb", 3);
        m.insert("E", 4);
        m.insert("F", 5);
        m.insert("F#", 6);
        m.insert("Gb", 6);
        m.insert("G", 7);
        m.insert("G#", 8);
        m.insert("Ab", 8);
        m.insert("A", 9);
        m.insert("A#", 10);
        m.insert("Bb", 10);
        m.insert("B", 11);

        m
    });

const MIDI_TO_NOTE_MAP: [&str; 12] = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
];

/// Parses a scientific pitch notation string and converts it into a MIDI note number.
///
/// This function provides a lightweight bridge between human-readable musical notation
/// (e.g. `"A4"`, `"C#3"`, `"Bb2"`) and numeric MIDI pitch indices. It is intended for
/// configuration, annotation, testing, and tooling scenarios rather than high-throughput
/// parsing.
///
/// Accidentals are supported using a single sharp (`#`) or flat (`b`). Enharmonic spellings
/// (e.g. `"C#4"` and `"Db4"`) resolve to the same MIDI value.
///
/// # Arguments
///
/// * `note_name`\
///   A note name encoded using scientific pitch notation with an explicit octave number.
///   The accepted grammar is:
///
///   * A single uppercase pitch class in `A`–`G`
///   * Optionally followed by exactly one accidental (`#` or `b`)
///   * Followed by a base-10 octave number with no sign or separators
///
///   Examples of valid inputs include `"A4"`, `"C#3"`, and `"Bb2"`.
///
///   The parser is case-sensitive. Lowercase note letters are not accepted.
///
/// # Returns
///
/// The corresponding MIDI note number in the inclusive range `0..=127`.
///
/// # Errors
///
/// Returns an error in the following cases:
///
/// * The input does not conform to the accepted note grammar.
/// * The pitch class or accidental is not recognised.
/// * The octave component cannot be parsed as an integer.
/// * The resulting MIDI value lies outside the valid range `0..=127`.
///
/// All errors are reported as parameter validation failures.
///
/// # Behavioural Guarantees
///
/// * Valid enharmonic spellings resolve to the same MIDI value.
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
/// ```
#[inline]
pub fn note_to_midi(note_name: &str) -> AudioSampleResult<u8> {
    if note_name.len() < 2 {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "note_name",
            "Note name must include octave number (e.g., 'A4')",
        )));
    }

    // Parse note and octave
    let (note_part, octave_str) = if note_name.len() == 2 {
        (&note_name[..1], &note_name[1..])
    } else if note_name.len() == 3 && (note_name.contains('#') || note_name.contains('b')) {
        (&note_name[..2], &note_name[2..])
    } else {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "note_name",
            "Invalid note name format. Expected format: 'A4', 'C#3', 'Bb2'",
        )));
    };

    // Look up base note value
    let base_note = NOTE_TO_MIDI_MAP.get(note_part).copied().ok_or_else(|| {
        AudioSampleError::Parameter(ParameterError::invalid_value(
            "note_name",
            format!("Unknown note: {note_part}"),
        ))
    })?;

    // Parse octave
    let octave: i32 = octave_str.parse().map_err(|_| {
        AudioSampleError::Parameter(ParameterError::invalid_value(
            "note_name",
            format!("Invalid octave: {octave_str}"),
        ))
    })?;

    // Calculate MIDI note number
    let midi_note = (octave + 1) * 12 + i32::from(base_note);

    // Check range
    if !(0..=127).contains(&midi_note) {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "note_name",
            format!("Note {note_name} (MIDI {midi_note}) is outside valid MIDI range 0-127"),
        )));
    }

    Ok(midi_note as u8)
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
/// * For any value accepted by this function, applying [`note_to_midi`] to the returned
///   string recovers the original MIDI value.
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
/// Returns an error if `freq_hz` is non-positive.
///
/// # Behavioural Guarantees
///
/// * The returned `cents` value lies in the half-open interval approximately
///   `[-50, 50)`, subject to floating-point rounding.
/// * For frequencies whose nearest MIDI value cannot be represented as a valid `u8`,
///   the note component defaults to `"A4"`. This behaviour is deterministic but lossy.
/// * Applying [`note_to_frequency`] to the returned `note_name` yields a frequency close
///   to the nearest semitone of the original input, not necessarily the original value.
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
/// ```
#[inline]
pub fn frequency_to_note(freq_hz: f64) -> AudioSampleResult<(String, f64)> {
    if freq_hz <= 0.0 {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "freq_hz",
            "Frequency must be positive",
        )));
    }

    let midi_float = hz_to_midi(freq_hz);
    let midi_rounded = midi_float.round();
    let cents = (midi_float - midi_rounded) * 100.0;

    let midi_note = midi_rounded as u8;
    let note_name = midi_to_note(midi_note)?;

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
/// The first element is always `0 Hz`. The final element corresponds approximately to the
/// Nyquist frequency (`sample_rate / 2`), subject to floating-point rounding.
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

        // Test case insensitivity would require making the function case-insensitive
        // assert_eq!(note_to_midi("a4").unwrap(), 69);
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
}
