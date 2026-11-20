//! The alternate `Display` form of multi-channel audio iterates over the
//! channels and does not panic for buffers with more samples than channels.

use audio_samples::{AudioSamples, sample_rate};
use ndarray::array;

#[test]
fn multichannel_display_lists_each_channel_once() {
    let audio: AudioSamples<'static, f32> = AudioSamples::new_multi_channel(
        array![[0.1f32, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
        sample_rate!(16000),
    )
    .unwrap();
    let text = format!("{audio:#}");
    assert!(text.contains("ch 0:"), "{text}");
    assert!(text.contains("ch 1:"), "{text}");
    assert!(!text.contains("ch 2:"), "{text}");
    let short = format!("{audio}");
    assert!(
        short.starts_with("AudioSamples<f32>: 2 ch × 4 samples"),
        "{short}"
    );
    assert!(!short.contains('\n'), "{short}");
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

fn mono<T: audio_samples::StandardSample + 'static>(samples: Vec<T>) -> AudioSamples<'static, T> {
    AudioSamples::new_mono(ndarray::Array1::from_vec(samples), sample_rate!(8000)).unwrap()
}

/// Integer arithmetic saturates in both build profiles instead of overflowing the sample
/// type (which would panic in a debug build and wrap in a release build).
#[test]
fn integer_arithmetic_saturates() {
    let a = mono(vec![30000i16, -30000, 100]);
    assert_eq!(
        (a.clone() + a.clone()).as_slice().unwrap(),
        &[32767, -32768, 200]
    );
    assert_eq!(
        (a.clone() + 10000).as_slice().unwrap(),
        &[32767, -20000, 10100]
    );
    assert_eq!(
        (a.clone() - 10000).as_slice().unwrap(),
        &[20000, -32768, -9900]
    );
    assert_eq!(
        (a.clone() * a.clone()).as_slice().unwrap(),
        &[32767, 32767, 10000]
    );

    let wide = mono(vec![i32::MAX, i32::MIN]);
    assert_eq!(
        (wide.clone() + wide.clone()).as_slice().unwrap(),
        &[i32::MAX, i32::MIN]
    );
    assert_eq!((-wide).as_slice().unwrap(), &[-i32::MAX, i32::MAX]);

    let top = audio_samples::I24::saturating_from_i32(8_388_607);
    let sum = mono(vec![top]) + mono(vec![top]);
    assert_eq!(sum.as_slice().unwrap()[0].to_i32(), 8_388_607);

    let mut in_place = mono(vec![30000i16]);
    in_place += mono(vec![30000i16]);
    assert_eq!(in_place.as_slice().unwrap(), &[32767]);
}

#[test]
fn integer_division_by_zero_gives_the_limit() {
    let a = mono(vec![100i16, -100, 0, 9]);
    let b = mono(vec![0i16, 0, 0, 2]);
    assert_eq!((a / b).as_slice().unwrap(), &[32767, -32768, 0, 4]);
}

/// `u8` samples are offset binary: 128 is silence.
#[test]
fn u8_arithmetic_respects_the_offset() {
    let a = mono(vec![128u8, 138, 255, 0]);
    let silence = mono(vec![128u8; 4]);
    assert_eq!(
        (a.clone() + silence).as_slice().unwrap(),
        a.as_slice().unwrap()
    );
    assert_eq!((a.clone() - a.clone()).as_slice().unwrap(), &[128; 4]);
    assert_eq!((a.clone() + a).as_slice().unwrap(), &[128, 148, 255, 0]);
}

#[test]
fn float_arithmetic_is_plain() {
    let a = mono(vec![0.5f32, -0.25]);
    assert_eq!((a.clone() + a.clone()).as_slice().unwrap(), &[1.0, -0.5]);
    assert_eq!((a.clone() * a.clone()).as_slice().unwrap(), &[0.25, 0.0625]);
    assert_eq!((a.clone() / a.clone()).as_slice().unwrap(), &[1.0, 1.0]);
    assert_eq!((-a).as_slice().unwrap(), &[-0.5, 0.25]);
}
