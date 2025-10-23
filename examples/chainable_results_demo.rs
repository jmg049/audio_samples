//! Demonstration of chainable result types for improved error handling ergonomics.
//!
//! This example shows how the ChainableResult type enables fluent, functional-style
//! error handling that reduces friction when chaining fallible audio operations.

use audio_samples::{AudioSamples, ChainableResult, IntoChainable};
use ndarray::Array1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Chainable Results Demo");
    println!("=====================");

    // Create test audio data
    let audio_data = Array1::from(vec![0.1f32, 0.5, -0.3, 0.8, -0.9, 0.2]);
    let mut audio = AudioSamples::new_mono(audio_data, 44100);

    println!(
        "Original audio: {} samples at {}Hz",
        audio.samples_per_channel(),
        audio.sample_rate()
    );

    // Example 1: Basic method chaining with error handling
    println!("\n1. Basic method chaining:");

    let result = audio
        .try_apply(|sample| sample * 0.5)
        .chain(|_| audio.try_apply(|sample| sample.clamp(-1.0, 1.0)))
        .map(|_| audio.samples_per_channel())
        .inspect(|count| println!("   Processed {} samples successfully", count))
        .log_on_error("Processing failed")
        .into_result();

    match result {
        Ok(sample_count) => println!("   ✓ Successfully processed {} samples", sample_count),
        Err(e) => println!("   ✗ Error: {}", e),
    }

    // Example 2: Type conversion with chainable results
    println!("\n2. Type conversion chain:");

    let conversion_result = audio
        .clone()
        .try_convert::<i16>()
        .map(|converted| {
            println!(
                "   Converted to i16, sample rate: {}",
                converted.sample_rate()
            );
            converted
        })
        .chain(|i16_audio| {
            // Chain another conversion
            i16_audio.try_convert::<f32>()
        })
        .inspect(|final_audio| {
            println!(
                "   Final audio: {} samples",
                final_audio.samples_per_channel()
            );
        })
        .into_result();

    match conversion_result {
        Ok(_) => println!("   ✓ Round-trip conversion successful"),
        Err(e) => println!("   ✗ Conversion error: {}", e),
    }

    // Example 3: Complex processing pipeline with graceful error handling
    println!("\n3. Complex processing pipeline:");

    let complex_pipeline =
        |mut audio_input: AudioSamples<f32>| -> ChainableResult<AudioSamples<f32>> {
            audio_input
                .try_apply(|sample| sample * 0.8) // Scale down
                .chain(|_| audio_input.try_apply(|sample| sample.clamp(-1.0, 1.0))) // Normalize
                .map(|_| audio_input.clone()) // Return processed audio
                .chain(|processed| {
                    // Chain validation
                    processed.try_validate()
                })
                .inspect(|final_audio| {
                    println!(
                        "   Pipeline complete: {}Hz, {} samples",
                        final_audio.sample_rate(),
                        final_audio.samples_per_channel()
                    );
                })
                .log_on_error("Pipeline processing failed")
        };

    let pipeline_result = complex_pipeline(audio.clone()).into_result();

    match pipeline_result {
        Ok(processed_audio) => {
            println!("   ✓ Complex pipeline successful");
            println!(
                "     Final sample rate: {}Hz",
                processed_audio.sample_rate()
            );
            println!(
                "     Final sample count: {}",
                processed_audio.samples_per_channel()
            );
        }
        Err(e) => println!("   ✗ Pipeline error: {}", e),
    }

    // Example 4: Error recovery with unwrap_or_else
    println!("\n4. Error recovery patterns:");

    let recovery_example = ChainableResult::ok(audio.clone())
        .and_then(|audio| {
            // Simulate a failure condition
            Err(audio_samples::AudioSampleError::InvalidParameter(
                "Simulated error".to_string(),
            ))
        })
        .map(|audio: AudioSamples<f32>| audio.samples_per_channel())
        .unwrap_or_else(|error| {
            println!("   Recovered from error: {}", error);
            0 // Default value
        });

    println!("   Recovery result: {} samples", recovery_example);

    // Example 5: Functional-style validation and processing
    println!("\n5. Functional validation chain:");

    let validate_and_process = |sample_rate: u32| -> ChainableResult<String> {
        ChainableResult::ok(sample_rate)
            .and_then(|rate| {
                if rate < 8000 || rate > 192000 {
                    Err(audio_samples::AudioSampleError::InvalidParameter(format!(
                        "Invalid sample rate: {}",
                        rate
                    )))
                } else {
                    Ok(rate)
                }
            })
            .map(|rate| rate * 2) // Double it
            .map(|doubled_rate| format!("Validated and doubled rate: {}Hz", doubled_rate))
            .inspect(|msg| println!("   {}", msg))
    };

    let valid_case = validate_and_process(44100).into_result();
    let invalid_case = validate_and_process(300000).into_result();

    match valid_case {
        Ok(msg) => println!("   ✓ Valid: {}", msg),
        Err(e) => println!("   ✗ Error: {}", e),
    }

    match invalid_case {
        Ok(msg) => println!("   ✓ Valid: {}", msg),
        Err(e) => println!("   ✗ Expected error: {}", e),
    }

    println!("\n✓ Chainable results demo completed!");
    println!("\nKey benefits of ChainableResult:");
    println!("  • Fluent method chaining for fallible operations");
    println!("  • Improved ergonomics over traditional Result<T, E>");
    println!("  • Built-in logging and inspection capabilities");
    println!("  • Seamless conversion to/from AudioSampleResult");
    println!("  • Functional programming patterns for error handling");

    Ok(())
}
