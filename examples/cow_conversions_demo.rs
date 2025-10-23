//! Demonstration of Cow-based conversions for efficient type handling.

use audio_samples::{AudioSamples, AudioStatistics};
use i24::I24;
use ndarray::array;
use std::borrow::Cow;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Cow-based Conversions Demo - Smart Memory Management");

    // Create some sample audio data
    let data = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
    let audio_f32 = AudioSamples::new_mono(data, 44100);

    println!(
        "Original f32 audio: {} samples",
        audio_f32.samples_per_channel()
    );

    // Demo 1: Same-type conversion (should borrow)
    println!("\n=== Same Type Conversion (f32 -> f32) ===");
    let cow_result = audio_f32.into_cow::<f32>()?;
    match cow_result {
        Cow::Borrowed(borrowed_audio) => {
            println!("✓ Zero-copy borrowed conversion!");
            println!("  Peak: {:.3}", borrowed_audio.peak());
            println!("  No memory allocation was needed");
        }
        Cow::Owned(_) => {
            println!("✗ Unexpected allocation for same type");
        }
    }

    // Demo 2: Different-type conversion (should own)
    println!("\n=== Different Type Conversion (f32 -> i16) ===");
    let cow_result_i16 = audio_f32.into_cow::<i16>()?;
    match cow_result_i16 {
        Cow::Borrowed(_) => {
            println!("✗ Unexpected borrowing for different types");
        }
        Cow::Owned(owned_audio) => {
            println!("✓ Proper owned conversion for different types");
            println!("  Peak: {}", owned_audio.peak());
            println!("  New allocation was needed for type conversion");
        }
    }

    // Demo 3: Function that can accept either borrowed or owned data
    analyze_audio_smart(&audio_f32.into_cow::<f32>()?);
    analyze_audio_smart(&audio_f32.into_cow::<i16>()?);

    // Demo 4: Conditional conversions based on runtime decisions
    let target_format = "f32"; // Could come from user input
    match target_format {
        "f32" => {
            let cow_audio = audio_f32.into_cow::<f32>()?;
            println!("\n=== Runtime Choice: f32 ===");
            process_audio_efficiently(cow_audio);
        }
        "i16" => {
            let cow_audio = audio_f32.into_cow::<i16>()?;
            println!("\n=== Runtime Choice: i16 ===");
            process_audio_efficiently(cow_audio);
        }
        _ => println!("Unsupported format"),
    }

    println!("\nCow conversions demo completed successfully!");
    Ok(())
}

/// Example function that can work efficiently with either borrowed or owned data
fn analyze_audio_smart<T: audio_samples::AudioSample>(cow_audio: &Cow<'_, AudioSamples<T>>)
where
    i16: audio_samples::ConvertTo<T>,
    I24: audio_samples::ConvertTo<T>,
    i32: audio_samples::ConvertTo<T>,
    f32: audio_samples::ConvertTo<T>,
    f64: audio_samples::ConvertTo<T>,
    for<'a> AudioSamples<T>: audio_samples::AudioTypeConversion<T>,
{
    let audio = cow_audio.as_ref();
    println!("Smart analysis of {} audio:", std::any::type_name::<T>());
    println!("  Samples: {}", audio.samples_per_channel());
    println!("  Peak: {:?}", audio.peak());

    match cow_audio {
        Cow::Borrowed(_) => println!("  ✓ Zero-copy analysis"),
        Cow::Owned(_) => println!("  → Converted data analysis"),
    }
}

/// Example function that processes audio efficiently regardless of ownership
fn process_audio_efficiently<T: audio_samples::AudioSample>(cow_audio: Cow<'_, AudioSamples<T>>)
where
    i16: audio_samples::ConvertTo<T>,
    I24: audio_samples::ConvertTo<T>,
    i32: audio_samples::ConvertTo<T>,
    f32: audio_samples::ConvertTo<T>,
    f64: audio_samples::ConvertTo<T>,
    for<'a> AudioSamples<T>: audio_samples::AudioTypeConversion<T>,
{
    println!(
        "Processing {} audio efficiently...",
        std::any::type_name::<T>()
    );

    // This function takes ownership of the Cow, so it can work with both
    // borrowed and owned data without additional allocations
    let audio = cow_audio.into_owned();
    println!("  Final peak after processing: {:?}", audio.peak());
}
