use audio_samples::{AudioSampleIterators, AudioSamples, PaddingMode};
use ndarray::array;

fn main() {
    println!("=== Audio Samples Mutable Iterators Demo ===");
    println!(
        "This example demonstrates when to use mutable iterators vs the optimized apply() methods.\n"
    );

    // Create some test audio data
    let audio = AudioSamples::new_multi_channel(
        array![
            [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], // Left channel
            [0.5f32, 1.0, 1.5, 2.0, 2.5, 3.0]  // Right channel
        ],
        44100,
    );

    println!("Original audio:");
    println!("Left:  {:?}", audio.as_multi_channel().unwrap().row(0));
    println!("Right: {:?}", audio.as_multi_channel().unwrap().row(1));

    // ===================================================================
    // Use Case 1: Simple element-wise operations
    // ===================================================================
    println!("\n--- Use Case 1: Simple Element-wise Operations ---");
    println!("For simple operations like gain/attenuation, use apply() for best performance:");

    let mut audio1 = audio.clone();

    // ✅ RECOMMENDED: Use apply() for simple element-wise operations
    audio1.apply(|sample| sample * 0.8);

    println!("After apply(|s| s * 0.8):");
    println!("Left:  {:?}", audio1.as_multi_channel().unwrap().row(0));
    println!("Right: {:?}", audio1.as_multi_channel().unwrap().row(1));

    // ===================================================================
    // Use Case 2: Frame-wise processing (cross-channel operations)
    // ===================================================================
    println!("\n--- Use Case 2: Frame-wise Processing ---");
    println!("For operations that need to process samples across channels, use frames_mut():");

    let mut audio2 = audio.clone();

    // ✅ USE ITERATORS: For cross-channel operations like stereo panning
    for mut frame in audio2.frames_mut() {
        // Implement a simple stereo widener
        if frame.len() == 2 {
            let left = *frame.get_mut(0).unwrap();
            let right = *frame.get_mut(1).unwrap();

            // Create wider stereo image
            let mid = (left + right) / 2.0;
            let side = (left - right) / 2.0;

            *frame.get_mut(0).unwrap() = mid + side * 1.5; // Enhanced left
            *frame.get_mut(1).unwrap() = mid - side * 1.5; // Enhanced right
        }
    }

    println!("After stereo widening:");
    println!("Left:  {:?}", audio2.as_multi_channel().unwrap().row(0));
    println!("Right: {:?}", audio2.as_multi_channel().unwrap().row(1));

    // ===================================================================
    // Use Case 3: Channel-specific processing
    // ===================================================================
    println!("\n--- Use Case 3: Channel-specific Processing ---");
    println!("For different processing per channel, use channels_mut():");

    let mut audio3 = audio.clone();

    // ✅ USE ITERATORS: For channel-specific processing
    for (ch_idx, channel) in audio3.channels_mut().enumerate() {
        if ch_idx == 0 {
            // Apply high-pass effect to left channel
            for sample in channel.iter_mut() {
                *sample = sample.max(0.0) * 1.2; // Simple "high-pass" + boost
            }
        } else {
            // Apply low-pass effect to right channel
            for sample in channel.iter_mut() {
                *sample = (*sample * 0.7).clamp(-1.0, 1.0); // Gentle attenuation
            }
        }
    }

    println!("After channel-specific processing:");
    println!("Left:  {:?}", audio3.as_multi_channel().unwrap().row(0));
    println!("Right: {:?}", audio3.as_multi_channel().unwrap().row(1));

    // ===================================================================
    // Use Case 4: Window-based processing
    // ===================================================================
    println!("\n--- Use Case 4: Window-based Processing ---");
    println!("For block-based operations like windowing or overlap-add, use windows_mut():");

    let mut audio4 =
        AudioSamples::new_mono(array![1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 44100);

    println!("Before windowing: {:?}", audio4.as_mono().unwrap());

    // ✅ USE ITERATORS: For windowed operations
    for mut window in audio4.windows_mut(4, 4) {
        // Non-overlapping 4-sample windows
        // Apply Hann window function
        window.apply_window_function(|i, window_size| {
            0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (window_size - 1) as f32).cos())
        });
    }

    println!("After Hann windowing: {:?}", audio4.as_mono().unwrap());

    // ===================================================================
    // Use Case 5: Complex frame processing with position information
    // ===================================================================
    println!("\n--- Use Case 5: Position-dependent Processing ---");
    println!(
        "For operations that need sample position, use apply_with_index() or frame iterators:"
    );

    let mut audio5 = audio.clone();

    // Option A: Use apply_with_index for position-dependent operations
    audio5.apply_with_index(|index, sample| {
        let fade_length = 3; // Fade over first 3 samples
        let gain = if index < fade_length {
            index as f32 / fade_length as f32
        } else {
            1.0
        };
        sample * gain
    });

    println!("After fade-in using apply_with_index():");
    println!("Left:  {:?}", audio5.as_multi_channel().unwrap().row(0));
    println!("Right: {:?}", audio5.as_multi_channel().unwrap().row(1));

    // ===================================================================
    // Performance Comparison
    // ===================================================================
    println!("\n--- Performance Guidelines ---");
    println!("📊 Performance ranking (fastest to slowest):");
    println!(
        "1. ✅ audio.apply(|s| s * gain)                    // Optimized ndarray mapv_inplace"
    );
    println!("2. ✅ audio.apply_with_index(|i, s| ...)           // Indexed operations");
    println!("3. 🔶 audio.channels_mut().enumerate()             // Channel-wise processing");
    println!("4. 🔶 audio.frames_mut()                           // Frame-wise processing");
    println!("5. 🔶 audio.windows_mut(size, hop)                 // Windowed processing");
    println!();
    println!("🎯 Use iterators when you need:");
    println!("   • Cross-channel operations (panning, stereo effects)");
    println!("   • Channel-specific processing (different EQ per channel)");
    println!("   • Block/window-based operations (STFT, overlap-add)");
    println!("   • Complex frame-level logic");
    println!();
    println!("⚡ Use apply() methods when you need:");
    println!("   • Simple element-wise operations (gain, filters)");
    println!("   • Maximum performance for large datasets");
    println!("   • Operations that work on individual samples");

    // ===================================================================
    // Advanced Example: Implementing a simple compressor using frames
    // ===================================================================
    println!("\n--- Advanced Example: Simple Compressor ---");

    let mut audio_comp = AudioSamples::new_multi_channel(
        array![
            [0.1f32, 0.8, 0.9, 0.3, 0.7, 0.95], // Dynamic content
            [0.2f32, 0.6, 0.85, 0.4, 0.75, 0.9]
        ],
        44100,
    );

    let threshold = 0.7;
    let ratio = 4.0;

    println!("Before compression:");
    println!("Left:  {:?}", audio_comp.as_multi_channel().unwrap().row(0));
    println!("Right: {:?}", audio_comp.as_multi_channel().unwrap().row(1));

    // Implement a simple compressor using frame-wise processing
    for mut frame in audio_comp.frames_mut() {
        // Calculate frame RMS
        let rms: f32 = frame.len() as f32;
        let mut sum_squares = 0.0;

        for ch in 0..frame.len() {
            if let Some(sample) = frame.get_mut(ch) {
                sum_squares += *sample * *sample;
            }
        }
        let frame_rms = (sum_squares / rms).sqrt();

        // Apply compression if above threshold
        if frame_rms > threshold {
            let excess = frame_rms - threshold;
            let compressed_excess = excess / ratio;
            let target_rms = threshold + compressed_excess;
            let gain = target_rms / frame_rms;

            // Apply gain to all channels in frame
            frame.apply(|sample| sample * gain);
        }
    }

    println!("After compression:");
    println!("Left:  {:?}", audio_comp.as_multi_channel().unwrap().row(0));
    println!("Right: {:?}", audio_comp.as_multi_channel().unwrap().row(1));

    println!(
        "\n✅ Mutable iterators provide powerful, safe ways to implement complex audio processing!"
    );
}
