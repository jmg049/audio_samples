use audio_samples::AudioSamples;
use ndarray::Array2;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Benchmarking SIMD channel processing vs scalar processing");
    
    // Test configurations: (channels, samples_per_channel)
    let test_configs = vec![
        (2, 44100),     // Stereo, 1 second
        (6, 44100),     // 5.1 surround, 1 second  
        (8, 44100),     // 7.1 surround, 1 second
        (2, 441000),    // Stereo, 10 seconds
    ];
    
    for &(channels, samples_per_channel) in &test_configs {
        println!("\nTesting {} channels × {} samples per channel", channels, samples_per_channel);
        
        // Generate test data
        let mut data = vec![0.0f32; channels * samples_per_channel];
        for (i, sample) in data.iter_mut().enumerate() {
            let t = (i % samples_per_channel) as f32 / 44100.0;
            let ch = i / samples_per_channel;
            *sample = (2.0 * std::f32::consts::PI * (440.0 + 100.0 * ch as f32) * t).sin() * 0.5;
        }
        
        let arr = Array2::from_shape_vec((channels, samples_per_channel), data)?;
        
        // Test scalar channel processing (legacy)
        let scalar_time = {
            let mut audio = AudioSamples::new_multi_channel(arr.clone(), 44100);
            
            let start = Instant::now();
            
            // Apply different gain to each channel using scalar processing
            audio.apply_channels(|channel, samples| {
                let gain = 0.8 - (channel as f32 * 0.1); // Decrease gain per channel
                for sample in samples.iter_mut() {
                    *sample = *sample * gain;
                }
                Ok(())
            })?;
            
            start.elapsed()
        };
        
        // Test SIMD channel processing (vectorized)
        let simd_time = {
            let mut audio = AudioSamples::new_multi_channel(arr.clone(), 44100);
            
            let start = Instant::now();
            
            // Apply same gain operation using SIMD processing
            audio.apply_channels_simd(|channel| {
                let gain = 0.8 - (channel as f32 * 0.1);
                move |sample| sample * gain
            })?;
            
            start.elapsed()
        };
        
        // Test parallel SIMD processing if available
        #[cfg(feature = "parallel-processing")]
        let parallel_time = {
            let mut audio = AudioSamples::new_multi_channel(arr.clone(), 44100);
            
            let start = Instant::now();
            
            // Apply same operation using parallel SIMD
            audio.apply_channels_parallel_simd(|channel| {
                let gain = 0.8 - (channel as f32 * 0.1);
                move |sample| sample * gain
            })?;
            
            start.elapsed()
        };
        
        println!("Scalar processing:     {:8.2} µs", scalar_time.as_micros());
        println!("SIMD processing:       {:8.2} µs", simd_time.as_micros());
        
        #[cfg(feature = "parallel-processing")]
        println!("Parallel SIMD:         {:8.2} µs", parallel_time.as_micros());
        
        if simd_time < scalar_time {
            let speedup = scalar_time.as_nanos() as f64 / simd_time.as_nanos() as f64;
            println!("SIMD speedup:          {:.1}x", speedup);
        }
        
        #[cfg(feature = "parallel-processing")]
        if parallel_time < scalar_time {
            let parallel_speedup = scalar_time.as_nanos() as f64 / parallel_time.as_nanos() as f64;
            println!("Parallel SIMD speedup: {:.1}x", parallel_speedup);
        }
        
        // Calculate throughput
        let total_samples = channels * samples_per_channel;
        let simd_throughput = (total_samples as f64) / simd_time.as_secs_f64() / 1_000_000.0;
        println!("SIMD throughput:       {:.0} million samples/second", simd_throughput);
        
        // Show theoretical benefit for multi-channel
        println!("Theoretical SIMD benefit for {} channels: up to {}x with full vectorization", 
                channels, if channels >= 4 { 4 } else { channels });
    }
    
    Ok(())
}