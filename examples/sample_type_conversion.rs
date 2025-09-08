//! Sample Type Conversion Example
//!
//! This example demonstrates comprehensive type-safe conversions between all
//! supported audio sample formats: i16, I24, i32, f32, and f64.
//!
//! Key concepts covered:
//! - Mathematical precision in conversions
//! - Bit depth scaling and range mapping
//! - Round-trip conversion accuracy
//! - Edge cases and boundary conditions
//! - Performance considerations
//!
//! Run with: `cargo run --example sample_type_conversion`

use audio_samples::{
    AudioSamples, AudioSample, AudioTypeConversion, ConvertTo, I24,
    AudioSampleResult, operations::*,
};
use ndarray::Array1;

fn main() -> AudioSampleResult<()> {
    println!("🔄 Sample Type Conversion Example");
    println!("=================================\n");

    // Create test audio with known values to demonstrate conversion behavior
    let test_values_f32 = vec![
        -1.0f32,    // Minimum float value
        -0.5,       // Half negative range
        0.0,        // Silence
        0.5,        // Half positive range
        1.0,        // Maximum float value
        0.25,       // Quarter range
        -0.75,      // Three-quarter negative range
    ];
    
    let original_audio = AudioSamples::new_mono(
        Array1::from(test_values_f32.clone()),
        48000
    );

    println!("📊 Original Audio (f32):");
    print_audio_info(&original_audio, "f32");
    println!();

    // Convert to all supported integer formats
    println!("🔢 Integer Conversions:");
    println!("----------------------");

    // Convert to i16
    let audio_i16 = original_audio.as_type::<i16>()?;
    print_audio_info(&audio_i16, "i16");
    print_conversion_details(&test_values_f32, &audio_i16, "f32 → i16");
    println!();

    // Convert to I24 (24-bit)
    let audio_i24 = original_audio.as_type::<I24>()?;
    print_audio_info(&audio_i24, "I24");
    print_i24_conversion_details(&test_values_f32, &audio_i24, "f32 → I24");
    println!();

    // Convert to i32
    let audio_i32 = original_audio.as_type::<i32>()?;
    print_audio_info(&audio_i32, "i32");
    print_conversion_details(&test_values_f32, &audio_i32, "f32 → i32");
    println!();

    // Convert to f64
    println!("🔢 Float Conversions:");
    println!("--------------------");
    let audio_f64 = original_audio.as_type::<f64>()?;
    print_audio_info(&audio_f64, "f64");
    print_conversion_details(&test_values_f32, &audio_f64, "f32 → f64");
    println!();

    // Demonstrate round-trip conversions
    println!("🔄 Round-trip Conversion Analysis:");
    println!("----------------------------------");
    
    // f32 → i16 → f32
    let roundtrip_f32_i16 = audio_i16.as_type::<f32>()?;
    print_roundtrip_analysis(&original_audio, &roundtrip_f32_i16, "f32 → i16 → f32");
    
    // f32 → i32 → f32
    let roundtrip_f32_i32 = audio_i32.as_type::<f32>()?;
    print_roundtrip_analysis(&original_audio, &roundtrip_f32_i32, "f32 → i32 → f32");
    
    // f32 → I24 → f32
    let roundtrip_f32_i24 = audio_i24.as_type::<f32>()?;
    print_roundtrip_analysis(&original_audio, &roundtrip_f32_i24, "f32 → I24 → f32");
    println!();

    // Demonstrate bit depth scaling
    println!("📏 Bit Depth Scaling Analysis:");
    println!("------------------------------");
    demonstrate_bit_depth_scaling()?;
    println!();

    // Edge cases and boundary conditions
    println!("⚠️  Edge Case Analysis:");
    println!("-----------------------");
    demonstrate_edge_cases()?;
    println!();

    // Practical conversion scenarios
    println!("🎯 Practical Conversion Scenarios:");
    println!("----------------------------------");
    demonstrate_practical_scenarios()?;

    println!("✅ Conversion analysis complete!");
    println!("\nKey takeaways:");
    println!("  • All conversions maintain mathematical precision");
    println!("  • Integer conversions use bit-shift scaling for accuracy");
    println!("  • Float conversions normalize to [-1.0, 1.0] range");
    println!("  • I24 provides professional audio precision (24-bit)");
    println!("  • Round-trip conversions preserve signal characteristics");

    Ok(())
}

/// Print basic information about an audio sample
fn print_audio_info<T>(audio: &AudioSamples<T>, type_name: &str) 
where
    T: AudioSample + std::fmt::Display,
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    AudioSamples<T>: AudioTypeConversion<T> + AudioStatistics<T>,
{
    println!("  Type: {} | Samples: {} | Peak: {}", 
             type_name, 
             audio.samples_per_channel(),
             audio.peak());
}

/// Print detailed conversion information for numeric types
fn print_conversion_details<T: AudioSample + std::fmt::Display>(
    original: &[f32], 
    converted: &AudioSamples<T>, 
    conversion_name: &str
) {
    println!("  {} conversions:", conversion_name);
    
    if let Some(mono_data) = converted.as_mono() {
        for (i, (&orig, &conv)) in original.iter().zip(mono_data.iter()).enumerate() {
            println!("    [{:}] {:.3} → {}", i, orig, conv);
        }
    }
}

/// Print detailed conversion information for I24 type
fn print_i24_conversion_details(
    original: &[f32], 
    converted: &AudioSamples<I24>, 
    conversion_name: &str
) {
    println!("  {} conversions:", conversion_name);
    
    if let Some(mono_data) = converted.as_mono() {
        for (i, (&orig, &conv)) in original.iter().zip(mono_data.iter()).enumerate() {
            println!("    [{:}] {:.3} → {} (i32: {})", i, orig, conv, conv.to_i32());
        }
    }
}

/// Analyze round-trip conversion accuracy
fn print_roundtrip_analysis<T: AudioSample>(
    original: &AudioSamples<f32>,
    roundtrip: &AudioSamples<T>,
    conversion_path: &str
) -> AudioSampleResult<()> 
where
    T: std::fmt::Display,
{
    println!("  {}:", conversion_path);
    
    let orig_mono = original.as_mono().unwrap();
    let rt_mono = roundtrip.as_mono().unwrap();
    
    let mut max_error = 0.0f32;
    let mut total_error = 0.0f32;
    
    for (i, (&orig, &rt)) in orig_mono.iter().zip(rt_mono.iter()).enumerate() {
        let orig_f32 = orig;
        let rt_f32: f32 = rt.convert_to()?;
        let error = (orig_f32 - rt_f32).abs();
        
        max_error = max_error.max(error);
        total_error += error;
        
        if i < 3 { // Show first few conversions
            println!("    [{:}] {:.6} → {:.6} (error: {:.6})", 
                     i, orig_f32, rt_f32, error);
        }
    }
    
    let avg_error = total_error / orig_mono.len() as f32;
    println!("    Max error: {:.6}, Avg error: {:.6}", max_error, avg_error);
    
    Ok(())
}

/// Demonstrate how bit depth affects dynamic range
fn demonstrate_bit_depth_scaling() -> AudioSampleResult<()> {
    // Create audio with maximum values for each type
    let max_values = vec![1.0f32];
    let _max_audio = AudioSamples::new_mono(Array1::from(max_values), 48000);
    
    println!("  Maximum value conversions (1.0 f32):");
    
    let max_i16: i16 = 1.0f32.convert_to()?;
    let max_i24: I24 = 1.0f32.convert_to()?;
    let max_i32: i32 = 1.0f32.convert_to()?;
    
    println!("    f32 1.0 → i16: {} ({})", max_i16, max_i16);
    println!("    f32 1.0 → I24: {} (i32: {})", max_i24, max_i24.to_i32());
    println!("    f32 1.0 → i32: {} ({})", max_i32, max_i32);
    
    // Show bit utilization
    println!("  Bit utilization:");
    println!("    i16: {} / {} ({:.1}%)", max_i16, i16::MAX, 
             (max_i16 as f32 / i16::MAX as f32) * 100.0);
    println!("    I24: {} / {} ({:.1}%)", max_i24.to_i32(), I24::MAX.to_i32(),
             (max_i24.to_i32() as f32 / I24::MAX.to_i32() as f32) * 100.0);
    println!("    i32: {} / {} ({:.1}%)", max_i32, i32::MAX,
             (max_i32 as f64 / i32::MAX as f64) * 100.0);

    Ok(())
}

/// Test edge cases and boundary conditions
fn demonstrate_edge_cases() -> AudioSampleResult<()> {
    println!("  Testing boundary values:");
    
    // Test extreme values
    let extreme_values = vec![-1.0f32, 1.0f32, 0.0f32];
    
    for &value in &extreme_values {
        println!("    f32 {} conversions:", value);
        
        let i16_val: i16 = value.convert_to()?;
        let i24_val: I24 = value.convert_to()?;
        let i32_val: i32 = value.convert_to()?;
        
        println!("      → i16: {}", i16_val);
        println!("      → I24: {} (i32: {})", i24_val, i24_val.to_i32());
        println!("      → i32: {}", i32_val);
        
        // Test round-trip
        let rt_f32_from_i16: f32 = i16_val.convert_to()?;
        let rt_f32_from_i24: f32 = i24_val.convert_to()?;
        let rt_f32_from_i32: f32 = i32_val.convert_to()?;
        
        println!("      Round-trip errors:");
        println!("        i16: {:.6}", (value - rt_f32_from_i16).abs());
        println!("        I24: {:.6}", (value - rt_f32_from_i24).abs());
        println!("        i32: {:.6}", (value - rt_f32_from_i32).abs());
    }
    
    // Test values outside valid range (should be clamped)
    println!("  Testing out-of-range values (should be clamped):");
    let out_of_range = vec![2.0f32, -2.0f32, 5.0f32];
    
    for &value in &out_of_range {
        let clamped_i16: i16 = value.convert_to()?;
        println!("    f32 {} → i16: {} (clamped)", value, clamped_i16);
    }

    Ok(())
}

/// Show practical conversion scenarios
fn demonstrate_practical_scenarios() -> AudioSampleResult<()> {
    println!("  Scenario 1: CD Quality to High-Res conversion (i16 → f32)");
    // Simulate CD quality audio (16-bit) being converted to float for processing
    let cd_quality_values = vec![
        i16::MAX / 2,     // Moderate level
        -i16::MAX / 4,    // Quiet level
        0,                // Silence
        i16::MAX,         // Maximum level
    ];
    
    let cd_audio = AudioSamples::new_mono(Array1::from(cd_quality_values.clone()), 44100);
    let hires_audio = cd_audio.as_type::<f32>()?;
    
    println!("    Original i16 values: {:?}", cd_quality_values);
    if let Some(float_data) = hires_audio.as_mono() {
        println!("    Converted f32 values: {:.6?}", float_data.to_vec());
    }
    
    println!("  Scenario 2: Professional recording format (I24 → f64)");
    // Professional 24-bit audio converted to double precision for analysis
    let professional_values = vec![
        I24::MAX,
        I24::try_from_i32(I24::MAX.to_i32() / 2).unwrap(),
        I24::try_from_i32(0).unwrap(),
        I24::MIN,
    ];
    
    let pro_audio = AudioSamples::new_mono(Array1::from(professional_values), 96000);
    let analysis_audio = pro_audio.as_type::<f64>()?;
    
    println!("    I24 range: {} to {}", I24::MIN.to_i32(), I24::MAX.to_i32());
    if let Some(double_data) = analysis_audio.as_mono() {
        println!("    f64 converted: {:.8?}", double_data.to_vec());
    }
    
    println!("  Scenario 3: Chain of conversions (simulating audio pipeline)");
    // Simulate a typical audio processing pipeline with multiple conversions
    let pipeline_start = vec![0.707f32, -0.5f32, 0.25f32]; // ~-3dB, -6dB, -12dB levels
    let original = AudioSamples::new_mono(Array1::from(pipeline_start.clone()), 48000);
    
    // Pipeline: f32 → i32 (recording) → f32 (processing) → i16 (output)
    let recorded = original.as_type::<i32>()?;
    let processed = recorded.as_type::<f32>()?;
    let output = processed.as_type::<i16>()?;
    let final_check = output.as_type::<f32>()?;
    
    println!("    Pipeline: f32 → i32 → f32 → i16 → f32");
    println!("    Original:  {:.6?}", pipeline_start);
    if let Some(final_data) = final_check.as_mono() {
        println!("    Final:     {:.6?}", final_data.to_vec());
        
        // Calculate total pipeline error
        let mut total_error = 0.0f32;
        for (orig, final_val) in pipeline_start.iter().zip(final_data.iter()) {
            total_error += (orig - final_val).abs();
        }
        println!("    Avg error: {:.6}", total_error / pipeline_start.len() as f32);
    }

    Ok(())
}