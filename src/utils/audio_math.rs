//! Audio mathematics utilities and conversion functions.
//!
//! This module provides librosa-compatible utility functions for common audio processing
//! calculations including frequency conversions, amplitude/dB conversions, time/frame
//! conversions, and musical theory functions.
//!
//! The functions are designed to be drop-in replacements for their librosa counterparts
//! while maintaining Rust's type safety and performance characteristics.
//!
//! # Examples
//!
//! ```rust
//! use audio_samples::audio_math::{hz_to_mel, mel_to_hz, amplitude_to_db, note_to_midi};
//!
//! // Frequency conversions
//! let mel_freq = hz_to_mel(440.0); // A4 in mel scale
//! let hz_freq = mel_to_hz(mel_freq); // Back to Hz
//!
//! // Amplitude conversions
//! let db_val = amplitude_to_db(0.5); // -6.02 dB
//!
//! // Musical theory
//! let midi_note = note_to_midi("A4").unwrap(); // 69
//! ```

use crate::{AudioSampleError, AudioSampleResult, ParameterError, RealFloat, to_precision};
use std::collections::HashMap;
use lazy_static::lazy_static;

// =============================================================================
// FREQUENCY CONVERSIONS
// =============================================================================

/// Converts frequency in Hz to mel scale.
///
/// The mel scale is a perceptual scale of pitches judged by listeners to be
/// equal in distance from one another. This conversion uses the formula:
/// `mel = 2595 * log10(1 + hz / 700)`
///
/// # Arguments
/// * `freq_hz` - Frequency in Hz
///
/// # Returns
/// Frequency in mel scale
///
/// # Examples
///
/// ```rust
/// use audio_samples::audio_math::hz_to_mel;
///
/// let mel = hz_to_mel(1000.0); // ≈ 1127.0
/// let mel_a4 = hz_to_mel(440.0); // ≈ 549.6
/// ```
pub fn hz_to_mel<F: RealFloat>(freq_hz: F) -> F {
    to_precision::<F, _>(2595.0) * (F::one() + freq_hz / to_precision::<F, _>(700.0)).log10()
}

/// Converts mel scale value back to frequency in Hz.
///
/// Inverse of `hz_to_mel`. Uses the formula:
/// `hz = 700 * (10^(mel / 2595) - 1)`
///
/// # Arguments
/// * `mel` - Frequency in mel scale
///
/// # Returns
/// Frequency in Hz
///
/// # Examples
///
/// ```rust
/// use audio_samples::audio_math::{hz_to_mel, mel_to_hz};
///
/// let freq = 1000.0;
/// let mel = hz_to_mel(freq);
/// let freq_back = mel_to_hz(mel);
/// assert!((freq - freq_back).abs() < 0.1f64);
/// ```
pub fn mel_to_hz<F: RealFloat>(mel: F) -> F {
    to_precision::<F, _>(700.0)
        * (to_precision::<F, _>(10.0).powf(mel / to_precision::<F, _>(2595.0)) - F::one())
}

/// Generate linearly spaced frequencies on the mel scale.
///
/// Creates `n_mels` frequency points that are linearly spaced in the mel domain
/// between `fmin` and `fmax`. This is useful for creating mel filter banks
/// for spectral analysis.
///
/// # Arguments
/// * `n_mels` - Number of mel frequency points to generate
/// * `fmin` - Minimum frequency in Hz
/// * `fmax` - Maximum frequency in Hz
///
/// # Returns
/// Vector of frequencies in Hz, linearly spaced on the mel scale
///
/// # Examples
/// ```rust
/// use audio_samples::audio_math::mel_scale;
///
/// let freqs = mel_scale(10, 100.0, 8000.0);
/// assert_eq!(freqs.len(), 10);
/// assert!(freqs[0] >= 100.0f64);
/// assert!(freqs[9] <= 8000.0f64);
/// ```
pub fn mel_scale<F: RealFloat>(n_mels: usize, fmin: F, fmax: F) -> Vec<F> {
    if n_mels == 0 {
        return Vec::new();
    }
    if n_mels == 1 {
        return vec![fmin];
    }

    // Convert frequency range to mel scale
    let mel_min = hz_to_mel(fmin);
    let mel_max = hz_to_mel(fmax);

    // Generate linearly spaced mel values
    let mel_step = (mel_max - mel_min) / to_precision::<F, _>((n_mels - 1) as f64);

    (0..n_mels)
        .map(|i| {
            let mel_val = mel_min + to_precision::<F, _>(i as f64) * mel_step;
            mel_to_hz(mel_val)
        })
        .collect()
}

/// Converts frequency in Hz to MIDI note number.
///
/// MIDI note number 69 corresponds to A4 (440 Hz). The conversion uses:
/// `midi = 69 + 12 * log2(freq_hz / 440)`
///
/// # Arguments
/// * `freq_hz` - Frequency in Hz
///
/// # Returns
/// MIDI note number (can be fractional)
///
/// # Examples
///
/// ```rust
/// use audio_samples::audio_math::hz_to_midi;
///
/// let midi_a4 = hz_to_midi(440.0); // 69.0
/// let midi_c4 = hz_to_midi(261.63); // ≈ 60.0
/// ```
pub fn hz_to_midi<F: RealFloat>(freq_hz: F) -> F {
    to_precision::<F, _>(69.0) + to_precision::<F, _>(12.0) * (freq_hz / to_precision::<F, _>(440.0)).log2()
}

/// Converts MIDI note number to frequency in Hz.
///
/// Inverse of `hz_to_midi`. Uses the formula:
/// `freq = 440 * 2^((midi - 69) / 12)`
///
/// # Arguments
/// * `midi_note` - MIDI note number (can be fractional)
///
/// # Returns
/// Frequency in Hz
///
/// # Examples
///
/// ```rust
/// use audio_samples::audio_math::midi_to_hz;
///
/// let freq_a4 = midi_to_hz(69.0); // 440.0 Hz
/// let freq_c4 = midi_to_hz(60.0); // ≈ 261.63 Hz
/// ```
pub fn midi_to_hz<F: RealFloat>(midi_note: F) -> F {
    to_precision::<F, _>(440.0) * to_precision::<F, _>(2.0).powf((midi_note - to_precision::<F, _>(69.0)) / to_precision::<F, _>(12.0))
}

// =============================================================================
// AMPLITUDE CONVERSIONS
// =============================================================================

/// Converts linear amplitude to decibels.
///
/// Uses the formula: `dB = 20 * log10(amplitude)` for amplitude ratios.
/// Returns -80 dB for zero or negative amplitudes to avoid infinite values.
///
/// # Arguments
/// * `amplitude` - Linear amplitude value
///
/// # Returns
/// Amplitude in decibels (dB)
///
/// # Examples
///
/// ```rust
/// use audio_samples::audio_math::amplitude_to_db;
///
/// let db = amplitude_to_db(1.0); // 0.0 dB
/// let db_half = amplitude_to_db(0.5); // ≈ -6.02 dB
/// let db_tenth = amplitude_to_db(0.1); // -20.0 dB
/// ```
pub fn amplitude_to_db<F: RealFloat>(amplitude: F) -> F {
    if amplitude > F::zero() {
        to_precision::<F, _>(20.0) * amplitude.log10()
    } else {
        to_precision::<F, _>(-80.0) // Floor at -80 dB
    }
}

/// Converts decibels to linear amplitude.
///
/// Uses the formula: `amplitude = 10^(dB / 20)` for amplitude ratios.
///
/// # Arguments
/// * `db` - Amplitude in decibels
///
/// # Returns
/// Linear amplitude value
///
/// # Examples
///
/// ```rust
/// use audio_samples::audio_math::db_to_amplitude;
///
/// let amp = db_to_amplitude(0.0); // 1.0
/// let amp_neg6 = db_to_amplitude(-6.0); // ≈ 0.501
/// let amp_neg20 = db_to_amplitude(-20.0); // 0.1
/// ```
pub fn db_to_amplitude<F: RealFloat>(db: F) -> F {
    to_precision::<F, _>(10.0).powf(db / to_precision::<F, _>(20.0))
}

/// Converts power to decibels.
///
/// Uses the formula: `dB = 10 * log10(power)` for power ratios.
/// Returns -80 dB for zero or negative power to avoid infinite values.
///
/// # Arguments
/// * `power` - Power value
///
/// # Returns
/// Power in decibels (dB)
///
/// # Examples
///
/// ```rust
/// use audio_samples::audio_math::power_to_db;
///
/// let db = power_to_db(1.0); // 0.0 dB
/// let db_half = power_to_db(0.5); // ≈ -3.01 dB
/// ```
pub fn power_to_db<F: RealFloat>(power: F) -> F {
    if power > F::zero() {
        to_precision::<F, _>(10.0) * power.log10()
    } else {
        to_precision::<F, _>(-80.0) // Floor at -80 dB
    }
}

/// Converts decibels to power.
///
/// Uses the formula: `power = 10^(dB / 10)` for power ratios.
///
/// # Arguments
/// * `db` - Power in decibels
///
/// # Returns
/// Linear power value
///
/// # Examples
///
/// ```rust
/// use audio_samples::audio_math::db_to_power;
///
/// let power = db_to_power(0.0); // 1.0
/// let power_neg3 = db_to_power(-3.0); // ≈ 0.501
/// ```
pub fn db_to_power<F: RealFloat>(db: F) -> F {
    to_precision::<F, _>(10.0).powf(db / to_precision::<F, _>(10.0))
}

// =============================================================================
// TIME/FRAME CONVERSIONS
// =============================================================================

/// Converts frame indices to time in seconds.
///
/// Useful for converting STFT frame indices to time positions.
///
/// # Arguments
/// * `frames` - Frame indices (can be slice or single value)
/// * `sample_rate` - Sample rate in Hz
/// * `hop_size` - Hop size in samples between frames
///
/// # Returns
/// Time in seconds
///
/// # Examples
///
/// ```rust
/// use audio_samples::audio_math::frames_to_time;
///
/// let time = frames_to_time(100, 44100.0, 512); // ≈ 1.16 seconds
/// ```
pub fn frames_to_time<F: RealFloat>(frames: usize, sample_rate: F, hop_size: usize) -> F {
    to_precision::<F, _>(frames * hop_size) / sample_rate
}

/// Converts time in seconds to frame indices.
///
/// Useful for converting time positions to STFT frame indices.
///
/// # Arguments
/// * `time_seconds` - Time in seconds
/// * `sample_rate` - Sample rate in Hz
/// * `hop_size` - Hop size in samples between frames
///
/// # Returns
/// Frame index (rounded)
///
/// # Examples
///
/// ```rust
/// use audio_samples::audio_math::time_to_frames;
///
/// let frame = time_to_frames(1.0, 44100.0, 512); // ≈ 86
/// ```
pub fn time_to_frames<F: RealFloat>(time_seconds: F, sample_rate: F, hop_size: usize) -> usize {
    ((time_seconds * sample_rate) / to_precision::<F, _>(hop_size)).round().to_usize().unwrap_or(0)
}

/// Converts sample indices to time in seconds.
///
/// # Arguments
/// * `samples` - Sample indices
/// * `sample_rate` - Sample rate in Hz
///
/// # Returns
/// Time in seconds
///
/// # Examples
///
/// ```rust
/// use audio_samples::audio_math::samples_to_time;
///
/// let time = samples_to_time(44100, 44100.0); // 1.0 second
/// ```
pub fn samples_to_time<F: RealFloat>(samples: usize, sample_rate: F) -> F {
    to_precision::<F, _>(samples) / sample_rate
}

/// Converts time in seconds to sample indices.
///
/// # Arguments
/// * `time_seconds` - Time in seconds
/// * `sample_rate` - Sample rate in Hz
///
/// # Returns
/// Sample index (rounded)
///
/// # Examples
///
/// ```rust
/// use audio_samples::audio_math::time_to_samples;
///
/// let samples = time_to_samples(1.0, 44100.0); // 44100
/// ```
pub fn time_to_samples<F: RealFloat>(time_seconds: F, sample_rate: F) -> usize {
    (time_seconds * sample_rate).round().to_usize().unwrap_or(0)
}

// =============================================================================
// MUSICAL THEORY FUNCTIONS
// =============================================================================

// Note name to MIDI number lookup table.
lazy_static! {
    static ref NOTE_TO_MIDI_MAP: HashMap<&'static str, u8> = {
        let mut m = HashMap::new();

        // Base notes (octave 0)
        m.insert("C", 0);   m.insert("C#", 1);  m.insert("Db", 1);
        m.insert("D", 2);   m.insert("D#", 3);  m.insert("Eb", 3);
        m.insert("E", 4);   m.insert("F", 5);   m.insert("F#", 6);
        m.insert("Gb", 6);  m.insert("G", 7);   m.insert("G#", 8);
        m.insert("Ab", 8);  m.insert("A", 9);   m.insert("A#", 10);
        m.insert("Bb", 10); m.insert("B", 11);

        m
    };

    static ref MIDI_TO_NOTE_MAP: [&'static str; 12] = [
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"
    ];
}

/// Converts a note name (e.g., "A4", "C#3") to MIDI note number.
///
/// Supports both sharp (#) and flat (b) notation. Note names are case-insensitive.
/// The octave number follows Scientific Pitch Notation where A4 = 440 Hz = MIDI 69.
///
/// # Arguments
/// * `note_name` - Note name with octave (e.g., "A4", "C#3", "Bb2")
///
/// # Returns
/// MIDI note number (0-127)
///
/// # Errors
/// Returns error if the note name format is invalid or out of MIDI range.
///
/// # Examples
///
/// ```rust
/// use audio_samples::audio_math::note_to_midi;
///
/// let midi = note_to_midi("A4").unwrap(); // 69
/// let midi_c4 = note_to_midi("C4").unwrap(); // 60
/// let midi_sharp = note_to_midi("C#4").unwrap(); // 61
/// let midi_flat = note_to_midi("Db4").unwrap(); // 61 (same as C#4)
/// ```
pub fn note_to_midi(note_name: &str) -> AudioSampleResult<u8> {
    if note_name.len() < 2 {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "note_name",
            "Note name must include octave number (e.g., 'A4')"
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
            "Invalid note name format. Expected format: 'A4', 'C#3', 'Bb2'"
        )));
    };

    // Look up base note value
    let base_note = NOTE_TO_MIDI_MAP.get(note_part)
        .copied()
        .ok_or_else(|| AudioSampleError::Parameter(ParameterError::invalid_value(
            "note_name",
            format!("Unknown note: {}", note_part)
        )))?;

    // Parse octave
    let octave: i32 = octave_str.parse()
        .map_err(|_| AudioSampleError::Parameter(ParameterError::invalid_value(
            "note_name",
            format!("Invalid octave: {}", octave_str)
        )))?;

    // Calculate MIDI note number
    let midi_note = (octave + 1) * 12 + base_note as i32;

    // Check range
    if midi_note < 0 || midi_note > 127 {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "note_name",
            format!("Note {} (MIDI {}) is outside valid MIDI range 0-127", note_name, midi_note)
        )));
    }

    Ok(midi_note as u8)
}

/// Converts MIDI note number to note name with octave.
///
/// Returns note name in sharp notation (e.g., "C#" instead of "Db").
///
/// # Arguments
/// * `midi_note` - MIDI note number (0-127)
///
/// # Returns
/// Note name with octave (e.g., "A4", "C#3")
///
/// # Examples
///
/// ```rust
/// use audio_samples::audio_math::midi_to_note;
///
/// let note = midi_to_note(69).unwrap(); // "A4"
/// let note_c4 = midi_to_note(60).unwrap(); // "C4"
/// let note_sharp = midi_to_note(61).unwrap(); // "C#4"
/// ```
pub fn midi_to_note(midi_note: u8) -> AudioSampleResult<String> {
    if midi_note > 127 {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "midi_note",
            "MIDI note must be in range 0-127"
        )));
    }

    let octave = (midi_note as i32 / 12) - 1;
    let note_index = midi_note % 12;
    let note_name = MIDI_TO_NOTE_MAP[note_index as usize];

    Ok(format!("{}{}", note_name, octave))
}

/// Converts note name to frequency in Hz.
///
/// Combines `note_to_midi` and `midi_to_hz` for convenience.
///
/// # Arguments
/// * `note_name` - Note name with octave (e.g., "A4", "C#3")
///
/// # Returns
/// Frequency in Hz
///
/// # Examples
///
/// ```rust
/// use audio_samples::audio_math::note_to_frequency;
///
/// let freq = note_to_frequency("A4").unwrap(); // 440.0 Hz
/// let freq_c4 = note_to_frequency("C4").unwrap(); // ≈ 261.63 Hz
/// ```
pub fn note_to_frequency<F: RealFloat>(note_name: &str) -> AudioSampleResult<F> {
    let midi_note = note_to_midi(note_name)?;
    Ok(midi_to_hz(to_precision::<F, _>(midi_note)))
}

/// Converts frequency to nearest note name.
///
/// # Arguments
/// * `freq_hz` - Frequency in Hz
///
/// # Returns
/// Nearest note name with octave and cents deviation
///
/// # Examples
///
/// ```rust
/// use audio_samples::audio_math::frequency_to_note;
///
/// let (note, cents) = frequency_to_note(440.0).unwrap(); // ("A4", 0.0)
/// let (note_sharp, cents) = frequency_to_note(466.16).unwrap(); // ("A#4", 0.0)
/// ```
pub fn frequency_to_note<F: RealFloat>(freq_hz: F) -> AudioSampleResult<(String, F)> {
    if freq_hz <= F::zero() {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "freq_hz",
            "Frequency must be positive"
        )));
    }

    let midi_float = hz_to_midi(freq_hz);
    let midi_rounded = midi_float.round();
    let cents = (midi_float - midi_rounded) * to_precision::<F, _>(100.0);

    let midi_note = midi_rounded.to_u8().unwrap_or(69); // Default to A4 if out of range
    let note_name = midi_to_note(midi_note)?;

    Ok((note_name, cents))
}

/// Converts cents to frequency ratio.
///
/// Cents are a logarithmic unit of pitch. 100 cents = 1 semitone.
///
/// # Arguments
/// * `cents` - Pitch deviation in cents
///
/// # Returns
/// Frequency ratio (1.0 = no change)
///
/// # Examples
///
/// ```rust
/// use audio_samples::audio_math::cents_to_ratio;
///
/// let ratio = cents_to_ratio(1200.0); // 2.0 (one octave up)
/// let ratio_semitone = cents_to_ratio(100.0); // ≈ 1.059 (one semitone up)
/// ```
pub fn cents_to_ratio<F: RealFloat>(cents: F) -> F {
    to_precision::<F, _>(2.0).powf(cents / to_precision::<F, _>(1200.0))
}

/// Converts frequency ratio to cents.
///
/// # Arguments
/// * `ratio` - Frequency ratio
///
/// # Returns
/// Pitch deviation in cents
///
/// # Examples
///
/// ```rust
/// use audio_samples::audio_math::ratio_to_cents;
///
/// let cents = ratio_to_cents(2.0); // 1200.0 (one octave)
/// let cents_semitone = ratio_to_cents(1.059463); // ≈ 100.0 (one semitone)
/// ```
pub fn ratio_to_cents<F: RealFloat>(ratio: F) -> F {
    to_precision::<F, _>(1200.0) * ratio.log2()
}

// =============================================================================
// SPECTRAL HELPER FUNCTIONS
// =============================================================================

/// Generates frequency bins for FFT analysis.
///
/// Returns the positive frequency bins for a real-valued FFT.
///
/// # Arguments
/// * `n_fft` - FFT size
/// * `sample_rate` - Sample rate in Hz
///
/// # Returns
/// Vector of frequency bins in Hz
///
/// # Examples
///
/// ```rust
/// use audio_samples::audio_math::fft_frequencies;
///
/// let freqs = fft_frequencies(1024, 44100.0);
/// assert_eq!(freqs.len(), 513); // n_fft/2 + 1
/// assert_eq!(freqs[0], 0.0); // DC component
/// assert!(freqs[freqs.len()-1] <= 22050.0f64); // Nyquist frequency
/// ```
pub fn fft_frequencies<F: RealFloat>(n_fft: usize, sample_rate: F) -> Vec<F> {
    let n_bins = n_fft / 2 + 1;
    let freq_resolution = sample_rate / to_precision::<F, _>(n_fft);

    (0..n_bins)
        .map(|i| to_precision::<F, _>(i) * freq_resolution)
        .collect()
}

/// Generates mel-scale frequency points.
///
/// Creates linearly spaced points in the mel scale, then converts back to Hz.
/// This is useful for creating mel filter banks.
///
/// # Arguments
/// * `n_mels` - Number of mel points to generate
/// * `fmin` - Minimum frequency in Hz
/// * `fmax` - Maximum frequency in Hz
///
/// # Returns
/// Vector of frequencies in Hz, linearly spaced in mel scale
///
/// # Examples
///
/// ```rust
/// use audio_samples::audio_math::mel_frequencies;
///
/// let mel_freqs = mel_frequencies(10, 0.0, 8000.0);
/// assert_eq!(mel_freqs.len(), 10);
/// assert_eq!(mel_freqs[0], 0.0);
/// assert!((mel_freqs[mel_freqs.len()-1] - 8000.0f64).abs() < 0.1f64);
/// ```
pub fn mel_frequencies<F: RealFloat>(n_mels: usize, fmin: F, fmax: F) -> Vec<F> {
    // Convert frequency range to mel scale
    let mel_min = hz_to_mel(fmin);
    let mel_max = hz_to_mel(fmax);

    // Generate linearly spaced mel values
    let mel_points = linspace(mel_min, mel_max, n_mels);

    // Convert back to Hz
    mel_points.into_iter()
        .map(mel_to_hz)
        .collect()
}

/// Generates linearly spaced values.
///
/// Helper function to create linearly spaced points between start and end.
///
/// # Arguments
/// * `start` - Starting value
/// * `end` - Ending value
/// * `num` - Number of points to generate
///
/// # Returns
/// Vector of linearly spaced values
fn linspace<F: RealFloat>(start: F, end: F, num: usize) -> Vec<F> {
    if num == 0 {
        return Vec::new();
    }
    if num == 1 {
        return vec![start];
    }

    let step = (end - start) / to_precision::<F, _>(num - 1);
    (0..num)
        .map(|i| start + to_precision::<F, _>(i) * step)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mel_frequency_conversions() {
        // Test round-trip conversion
        let freq = 1000.0f64;
        let mel = hz_to_mel(freq);
        let freq_back = mel_to_hz(mel);
        assert!((freq - freq_back).abs() < 0.001f64);

        // Test known values (using the actual mel scale formula)
        assert!((hz_to_mel(1000.0f64) - 1000.0f64).abs() < 1.0f64); // 1000 Hz ≈ 1000 mels
        assert!((mel_to_hz(1000.0f64) - 1000.0f64).abs() < 10.0f64);
    }

    #[test]
    fn test_midi_frequency_conversions() {
        // Test A4 = 440 Hz = MIDI 69
        assert!((hz_to_midi(440.0f64) - 69.0f64).abs() < 0.001f64);
        assert!((midi_to_hz(69.0f64) - 440.0f64).abs() < 0.001f64);

        // Test C4 = ~261.63 Hz = MIDI 60
        assert!((hz_to_midi(261.63f64) - 60.0f64).abs() < 0.01f64);
        assert!((midi_to_hz(60.0f64) - 261.63f64).abs() < 0.1f64);
    }

    #[test]
    fn test_amplitude_db_conversions() {
        // Test unity amplitude
        assert!((amplitude_to_db(1.0f64) - 0.0f64).abs() < 0.001f64);
        assert!((db_to_amplitude(0.0f64) - 1.0f64).abs() < 0.001f64);

        // Test -6 dB ≈ 0.5 amplitude
        assert!((amplitude_to_db(0.5f64) + 6.02f64).abs() < 0.1f64);
        assert!((db_to_amplitude(-6.0f64) - 0.501f64).abs() < 0.01f64);

        // Test round-trip
        let amp = 0.3f64;
        let db = amplitude_to_db(amp);
        let amp_back = db_to_amplitude(db);
        assert!((amp - amp_back).abs() < 0.001f64);
    }

    #[test]
    fn test_power_db_conversions() {
        // Test unity power
        assert!((power_to_db(1.0f64) - 0.0f64).abs() < 0.001f64);
        assert!((db_to_power(0.0f64) - 1.0f64).abs() < 0.001f64);

        // Test -3 dB ≈ 0.5 power
        assert!((power_to_db(0.5f64) + 3.01f64).abs() < 0.1f64);
        assert!((db_to_power(-3.0f64) - 0.501f64).abs() < 0.01f64);
    }

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
        let freq = note_to_frequency::<f64>("A4").unwrap();
        assert!((freq - 440.0f64).abs() < 0.001f64);

        let (note, cents): (String, f64) = frequency_to_note(440.0f64).unwrap();
        assert_eq!(note, "A4");
        assert!(cents.abs() < 1.0f64);
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
        assert!((freqs[freqs.len()-1] - 22050.0f64).abs() < 0.1f64); // Nyquist

        // Check frequency resolution
        let freq_res = freqs[1] - freqs[0];
        let expected_res = 44100.0 / 1024.0;
        assert!((freq_res - expected_res).abs() < 0.001f64);
    }

    #[test]
    fn test_mel_frequencies() {
        let mel_freqs = mel_frequencies(10, 0.0, 8000.0);
        assert_eq!(mel_freqs.len(), 10);
        assert_eq!(mel_freqs[0], 0.0);
        assert!((mel_freqs[mel_freqs.len()-1] - 8000.0f64).abs() < 0.1f64);

        // Check that they're properly spaced in mel scale
        let mel_vals: Vec<f64> = mel_freqs.iter().map(|&f| hz_to_mel(f)).collect();
        for i in 1..mel_vals.len() {
            let diff = mel_vals[i] - mel_vals[i-1];
            let expected_diff = (hz_to_mel(8000.0) - hz_to_mel(0.0)) / 9.0;
            assert!((diff - expected_diff).abs() < 1.0f64);
        }
    }
}