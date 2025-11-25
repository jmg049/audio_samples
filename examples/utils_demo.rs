//! Audio utility functions demonstration.
//!
//! This example showcases the various audio mathematics utility functions
//! available in the audio_samples library, demonstrating conversions between
//! different audio-related scales and measurements.

use audio_samples::{
    AudioSampleResult,
    // Re-exported audio math functions
    amplitude_to_db, db_to_amplitude, fft_frequencies, frames_to_time, hz_to_mel, hz_to_midi,
    mel_to_hz, mel_scale, midi_to_hz, midi_to_note, note_to_midi, power_to_db, time_to_frames,
};

fn main() -> AudioSampleResult<()> {
    println!("Audio Maths Utility Functions Demo");
    println!("==========================================");

    // Frequency conversion demonstrations
    frequency_conversions_demo()?;

    // Amplitude/dB conversion demonstrations
    amplitude_conversions_demo();

    // Musical theory demonstrations
    musical_theory_demo()?;

    // Time/frame conversion demonstrations
    time_frame_conversions_demo();

    // Spectral analysis helper demonstrations
    spectral_helpers_demo();

    Ok(())
}

fn frequency_conversions_demo() -> AudioSampleResult<()> {
    println!("\nFrequency Scale Conversions");
    println!("=============================");

    // Hz to Mel conversions
    println!("\nHz ↔ Mel Scale Conversions:");
    let frequencies_hz = [100.0f64, 440.0, 1000.0, 4000.0, 8000.0];

    for freq_hz in frequencies_hz {
        let mel = hz_to_mel(freq_hz);
        let hz_back = mel_to_hz(mel);
        println!("  {} Hz → {:.1} mels → {:.1} Hz (error: {:.3})",
                freq_hz, mel, hz_back, (freq_hz - hz_back).abs());
    }

    // Hz to MIDI conversions
    println!("\nHz ↔ MIDI Note Number Conversions:");
    let musical_frequencies = [
        (440.0f64, "A4"),      // Concert pitch
        (261.63, "C4"),     // Middle C
        (523.25, "C5"),     // C5
        (1046.5, "C6"),     // C6
        (82.41, "E2"),      // Low E (guitar)
    ];

    for (freq_hz, note_name) in musical_frequencies {
        let midi = hz_to_midi(freq_hz);
        let hz_back = midi_to_hz(midi);
        println!("  {} Hz ({}) → MIDI {:.1} → {:.2} Hz (error: {:.3})",
                freq_hz, note_name, midi, hz_back, (freq_hz - hz_back).abs());
    }

    // Mel scale demonstration
    println!("\nMel Scale Properties:");
    println!("  The mel scale provides perceptually uniform frequency spacing:");
    let mel_points = mel_scale(40, 80.0, 8000.0);
    println!("  Generated {} mel-spaced frequencies from 80 Hz to 8000 Hz:", mel_points.len());
    println!("  First 5: {:.1}, {:.1}, {:.1}, {:.1}, {:.1}",
            mel_points[0], mel_points[1], mel_points[2], mel_points[3], mel_points[4]);
    println!("  Last 5: {:.1}, {:.1}, {:.1}, {:.1}, {:.1}",
            mel_points[35], mel_points[36], mel_points[37], mel_points[38], mel_points[39]);

    Ok(())
}

fn amplitude_conversions_demo() {
    println!("\nAmplitude ↔ Decibel Conversions");
    println!("==================================");

    println!("\nLinear Amplitude to dB:");
    let amplitudes = [0.001f64, 0.1, 0.5, 1.0, 2.0, 10.0];

    for amp in amplitudes {
        let db = amplitude_to_db(amp);
        let amp_back = db_to_amplitude(db);
        println!("  {:.3} → {:.1} dB → {:.3} (error: {:.6})",
                amp, db, amp_back, (amp - amp_back).abs());
    }

    println!("\nPower to dB (10*log10):");
    let powers = [0.001f64, 0.1, 1.0, 10.0, 100.0];

    for power in powers {
        let db = power_to_db(power);
        println!("  Power {:.3} → {:.1} dB", power, db);
    }

    println!("\nCommon Audio Reference Levels:");
    println!("  • Unity gain (0 dB): {:.3}", db_to_amplitude(0.0f64));
    println!("  • -6 dB (half power): {:.3}", db_to_amplitude(-6.0f64));
    println!("  • -12 dB: {:.3}", db_to_amplitude(-12.0f64));
    println!("  • -20 dB: {:.3}", db_to_amplitude(-20.0f64));
    println!("  • -40 dB (very quiet): {:.6}", db_to_amplitude(-40.0f64));
    println!("  • -60 dB (noise floor): {:.9}", db_to_amplitude(-60.0f64));
}

fn musical_theory_demo() -> AudioSampleResult<()> {
    println!("\nMusical Theory Functions");
    println!("===========================");

    println!("\nNote Name ↔ MIDI Number Conversions:");
    let notes = ["C4", "C#4", "D4", "D#4", "E4", "F4", "F#4", "G4", "G#4", "A4", "A#4", "B4"];

    for note in notes {
        let midi = note_to_midi(note)?;
        let note_back = midi_to_note(midi)?;
        let freq = midi_to_hz(midi as f64);
        println!("  {} → MIDI {} → {} → {:.2} Hz",
                note, midi, note_back, freq);
    }

    println!("\n🎹 Piano Range Demonstration:");
    println!("  Piano typically spans MIDI notes 21-108:");
    let piano_notes = [21, 24, 36, 48, 60, 72, 84, 96, 108]; // A0, C1, C2, C3, C4, C5, C6, C7, C8

    for midi in piano_notes {
        let note = midi_to_note(midi)?;
        let freq = midi_to_hz(midi as f64);
        println!("  MIDI {} ({}) = {:.2} Hz", midi, note, freq);
    }

    println!("\n🎸 Guitar String Frequencies:");
    let guitar_tuning = [
        ("E2", "6th string (lowest)"),
        ("A2", "5th string"),
        ("D3", "4th string"),
        ("G3", "3rd string"),
        ("B3", "2nd string"),
        ("E4", "1st string (highest)"),
    ];

    for (note, description) in guitar_tuning {
        let midi = note_to_midi(note)?;
        let freq = midi_to_hz(midi as f64);
        println!("  {} (MIDI {}) = {:.2} Hz - {}", note, midi, freq, description);
    }

    Ok(())
}

fn time_frame_conversions_demo() {
    println!("\nTime ↔ Frame Conversions");
    println!("===========================");

    let sample_rate = 44100.0;
    let hop_size = 512;

    println!("\nFrame-based Analysis Parameters:");
    println!("  Sample rate: {} Hz", sample_rate);
    println!("  Hop size: {} samples", hop_size);
    println!("  Frame rate: {:.2} frames/second", sample_rate / hop_size as f64);
    println!("  Frame duration: {:.3} ms", (hop_size as f64 / sample_rate) * 1000.0);

    println!("\nFrames to Time Conversion:");
    let frame_numbers = [0, 10, 100, 1000, 4410]; // Various frame positions

    for frames in frame_numbers {
        let time_sec = frames_to_time::<f64>(frames, sample_rate, hop_size);
        let frames_back = time_to_frames(time_sec, sample_rate, hop_size);
        println!("  Frame {} → {:.3} sec → frame {} (error: {})",
                frames, time_sec, frames_back, (frames as isize - frames_back as isize).abs());
    }

    println!("\nTypical Analysis Durations:");
    let durations = [0.1, 0.5, 1.0, 5.0, 30.0]; // seconds

    for duration in durations {
        let frames = time_to_frames(duration, sample_rate, hop_size);
        println!("  {:.1} sec = {} frames = {} analysis windows",
                duration, frames, frames);
    }
}

fn spectral_helpers_demo() {
    println!("\nSpectral Analysis Helpers");
    println!("============================");

    // FFT frequency bin helpers
    println!("\nFFT Frequency Bins:");
    let sample_rate = 44100.0;
    let fft_sizes = [512, 1024, 2048, 4096];

    for fft_size in fft_sizes {
        let freqs = fft_frequencies(fft_size, sample_rate);
        let nyquist_bin = freqs.len() - 1;

        println!("\n  FFT size: {} samples", fft_size);
        println!("    Frequency resolution: {:.2} Hz/bin", freqs[1] - freqs[0]);
        println!("    Number of bins: {} (0 to {})", freqs.len(), nyquist_bin);
        println!("    DC bin (0): {:.1} Hz", freqs[0]);
        println!("    Nyquist bin ({}): {:.1} Hz", nyquist_bin, freqs[nyquist_bin]);

        // Find bins for common frequencies
        let target_freqs = [100.0, 440.0, 1000.0, 4000.0];
        println!("    Common frequency bins:");
        for target in target_freqs {
            let bin = (target / (sample_rate / fft_size as f64)) as usize;
            if bin < freqs.len() {
                println!("      {:.0} Hz ≈ bin {} ({:.1} Hz)", target, bin, freqs[bin]);
            }
        }
    }

    println!("\nFrequency Band Analysis:");
    println!("  Common audio frequency bands and their FFT bin ranges (for 2048-point FFT @ 44.1kHz):");

    let bands = [
        ("Sub-bass", 20.0, 60.0),
        ("Bass", 60.0, 250.0),
        ("Low midrange", 250.0, 500.0),
        ("Midrange", 500.0, 2000.0),
        ("Upper midrange", 2000.0, 4000.0),
        ("Presence", 4000.0, 6000.0),
        ("Brilliance", 6000.0, 20000.0),
    ];

    let freqs = fft_frequencies(2048, sample_rate);

    for (name, low_freq, high_freq) in bands {
        let low_bin = (low_freq / (sample_rate / 2048.0)) as usize;
        let high_bin = (high_freq / (sample_rate / 2048.0)) as usize;
        let high_bin = high_bin.min(freqs.len() - 1);

        println!("  • {}: {:.0}-{:.0} Hz → bins {}-{} ({} bins)",
                name, low_freq, high_freq, low_bin, high_bin, high_bin - low_bin + 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_utils_demo() {
        // Run the main demo as a test to ensure all functions work
        main().expect("Utils demo should complete without error");
    }

    #[test]
    fn test_round_trip_conversions() {
        // Test that all conversions are properly reversible

        // Frequency conversions
        let freq = 440.0f64;
        assert!((freq - mel_to_hz(hz_to_mel(freq))).abs() < 0.01);
        assert!((freq - midi_to_hz(hz_to_midi(freq))).abs() < 0.01);

        // Amplitude conversions
        let amp = 0.5f64;
        assert!((amp - db_to_amplitude(amplitude_to_db(amp))).abs() < 1e-10);

        // Note conversions
        let note = "A4";
        let midi = note_to_midi(note).unwrap();
        let note_back = midi_to_note(midi).unwrap();
        assert_eq!(note, note_back);

        // Time/frame conversions
        let frames = 1000;
        let sample_rate = 44100.0f64;
        let hop_size = 512;
        let time = frames_to_time::<f64>(frames, sample_rate, hop_size);
        let frames_back = time_to_frames(time, sample_rate, hop_size);
        assert_eq!(frames, frames_back);
    }

    #[test]
    fn test_spectral_functions() {
        // Test FFT frequency calculation
        let freqs = fft_frequencies(1024, 44100.0f64);
        assert_eq!(freqs.len(), 513); // N/2 + 1 for real FFT
        assert!((freqs[0] - 0.0f64).abs() < 1e-10); // DC
        assert!((freqs[512] - 22050.0f64).abs() < 1.0); // Nyquist (approximately)

        // Test mel scale generation
        let mel_freqs = mel_scale(40, 100.0f64, 8000.0f64);
        assert_eq!(mel_freqs.len(), 40);
        assert!((mel_freqs[0] - 100.0f64).abs() < 1.0);
        assert!((mel_freqs[39] - 8000.0f64).abs() < 10.0);
    }

    #[test]
    fn test_known_values() {
        // Test against known reference values

        // A4 = 440 Hz = MIDI note 69
        assert!((hz_to_midi(440.0f64) - 69.0f64).abs() < 0.1);
        assert!((midi_to_hz(69.0f64) - 440.0f64).abs() < 0.1);

        // 0 dB = unity gain = 1.0
        assert!((amplitude_to_db(1.0f64) - 0.0f64).abs() < 1e-10);
        assert!((db_to_amplitude(0.0f64) - 1.0f64).abs() < 1e-10);

        // -6 dB ≈ 0.5 (half amplitude)
        assert!((db_to_amplitude(-6.0f64) - 0.5012f64).abs() < 0.001);
    }
}