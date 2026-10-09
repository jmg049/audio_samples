//! Regression test: the short `Display` form of multi-channel audio iterated
//! over the sample count instead of the channel count and panicked with
//! "index < dim" for any buffer with more samples than channels.

use audio_samples::{AudioSamples, sample_rate};
use ndarray::array;

#[test]
fn multichannel_display_lists_each_channel_once() {
    let audio: AudioSamples<'static, f32> = AudioSamples::new_multi_channel(
        array![[0.1f32, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
        sample_rate!(16000),
    )
    .unwrap();
    let text = format!("{audio}");
    assert!(text.contains("Ch 0:"), "{text}");
    assert!(text.contains("Ch 1:"), "{text}");
    assert!(!text.contains("Ch 2:"), "{text}");
}
