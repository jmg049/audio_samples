#[cfg(not(feature = "editing"))]
fn main() {
    eprintln!("This example requires the 'editing' feature.");
}

#[cfg(feature = "editing")]
#[inline]
pub fn main() -> audio_samples::AudioSampleResult<()> {
    use audio_samples::operations::types::{FadeCurve, PadSide};
    use audio_samples::{AudioEditing, AudioSamples, AudioStatistics, sine_wave};
    use std::time::Duration;

    let sample_rate_hz = core::num::NonZeroU32::new(44_100).unwrap();

    // Base signal: 1 second sine.
    let audio: AudioSamples<'static, f64> =
        sine_wave::<f64>(440.0, Duration::from_secs(1), sample_rate_hz, 0.8);
    println!(
        "Original:  dur={:.3}s  peak={:.4}",
        audio.duration_seconds(),
        audio.peak()
    );

    // Trim a segment.
    let segment = audio.trim(0.2f64, 0.6f64)?;
    println!("Trim 0.2..0.6s: dur={:.3}s", segment.duration_seconds());

    // Pad and trim silence.
    let padded = audio.pad(0.25f64, 0.25f64, 0.0)?;
    println!("Padded: dur={:.3}s", padded.duration_seconds());
    let trimmed = padded.trim_silence(-60.0f64)?;
    println!("Trim silence: dur={:.3}s", trimmed.duration_seconds());

    // Fade in/out in-place.
    let mut faded = audio.clone();
    faded.fade_in(0.05f64, FadeCurve::SmoothStep)?;
    faded.fade_out(0.05f64, FadeCurve::SmoothStep)?;
    println!("Faded: peak={:.4}  rms={:.4}", faded.peak(), faded.rms());

    // Split + concatenate.
    let segments = audio.split(0.25f64)?;
    println!("Split into {} segments", segments.len());
    let reconstructed =
        AudioSamples::concatenate_owned(non_empty_slice::NonEmptyVec::new(segments).unwrap())?;
    println!("Concatenated: dur={:.3}s", reconstructed.duration_seconds());

    // Repeat.
    let repeated = audio.repeat(3)?;
    println!("Repeated x3: dur={:.3}s", repeated.duration_seconds());

    // Reverse.
    let reversed = audio.reverse();
    println!(
        "Reverse: first_sample={:.4}",
        reversed.as_mono().unwrap()[0]
    );

    // Pad to an exact duration.
    let to_2s = audio.pad_to_duration(2.0f64, 0.0, PadSide::Right)?;
    println!("Pad to 2s: dur={:.3}s", to_2s.duration_seconds());

    // Mix (static method on the trait).
    let detuned = sine_wave::<f64>(445.0, Duration::from_secs(1), sample_rate_hz, 0.8);
    let sources = [audio, detuned];
    let weights = [0.7f64, 0.3f64];
    let mixed = AudioSamples::mix(
        non_empty_slice::NonEmptySlice::from_slice(&sources).unwrap(),
        Some(non_empty_slice::NonEmptySlice::from_slice(&weights).unwrap()),
    )?;
    println!("Mixed: peak={:.4}  rms={:.4}", mixed.peak(), mixed.rms());

    Ok(())
}
