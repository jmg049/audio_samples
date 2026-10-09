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

#[test]
fn integer_means_do_not_overflow() {
    use audio_samples::AudioData;
    // 4 x 30000 in i16 sums to 120000, beyond i16::MAX.
    let mono: AudioSamples<'static, i16> =
        AudioSamples::new_mono(array![30000i16, 30000, 30000, 30000], sample_rate!(8000)).unwrap();
    let data: &AudioData<'_, i16> = mono.data();
    assert_eq!(data.mean(), 30000);
    let multi: AudioSamples<'static, i16> = AudioSamples::new_multi_channel(
        array![[30000i16, 30000], [30000, 30000]],
        sample_rate!(8000),
    )
    .unwrap();
    assert_eq!(multi.data().mean(), 30000);
}

#[test]
fn multichannel_variance_axis_follows_ndarray() {
    use audio_samples::AudioData;
    // Channel 0 = [1, 3] (variance 1), channel 1 = [2, 2] (variance 0).
    let audio: AudioSamples<'static, f64> =
        AudioSamples::new_multi_channel(array![[1.0f64, 3.0], [2.0, 2.0]], sample_rate!(8000))
            .unwrap();
    let AudioData::Multi(m) = audio.data() else {
        panic!("expected multi-channel data")
    };
    // Per channel (over samples): [1, 0].
    let per_channel = m.variance_axis_ddof(ndarray::Axis(1), 0);
    assert_eq!(per_channel.to_vec(), vec![1.0, 0.0]);
    // Per sample position (across channels): [(1-1.5)^2, (3-2.5)^2] = [0.25, 0.25].
    let per_sample = m.variance_axis_ddof(ndarray::Axis(0), 0);
    assert_eq!(per_sample.to_vec(), vec![0.25, 0.25]);
}
