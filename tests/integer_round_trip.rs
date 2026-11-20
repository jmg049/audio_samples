//! Integer sample types through audio operations.
//!
//! Every operation that returns audio gives back the input's sample type. For
//! integer types the work is done in `f64` and the result is converted back
//! with rounding and saturation. These tests check, for each such operation,
//! that the `i16` (and where relevant `I24`, `i32`, `u8`) result
//!
//! - has the input's sample type,
//! - matches the same operation run on the `f64` version of the audio and
//!   re-quantised with `to_format::<i16>()` to within 1 LSB, and
//! - rounds to nearest instead of truncating (exact values on small inputs).
//!
//! Float results are checked bit for bit against the operation's `f64` formula
//! cast to the sample type.
//!
//! Run with:
//! ```bash
//! cargo test --test integer_round_trip --no-default-features --features processing,channels
//! ```

use audio_samples::{
    AudioChannelOps, AudioProcessing, AudioSampleError, AudioSamples, AudioTypeConversion,
    ConvertFrom, ConvertTo, I24, NormalizationConfig, StandardSample,
    operations::types::{MonoConversionMethod, StereoConversionMethod},
    sample_rate,
};
use ndarray::{Array1, Array2, array};
use non_empty_slice::NonEmptySlice;

fn test_signal(n: usize) -> Array1<i16> {
    Array1::from_iter((0..n).map(|i| {
        let t = i as f64;
        (12000.0 * (0.05 * t).sin() + 7000.0 * (0.31 * t).sin() + 1234.0).round() as i16
    }))
}

fn test_signal_f32(n: usize) -> Array1<f32> {
    Array1::from_iter((0..n).map(|i| {
        let t = i as f32;
        0.4 * (0.05 * t).sin() + 0.2 * (0.31 * t).sin() + 0.037
    }))
}

fn stereo_i16() -> AudioSamples<'static, i16> {
    let l = test_signal(300);
    let r = test_signal(301).slice(ndarray::s![1..]).to_owned();
    let mut m = Array2::<i16>::zeros((2, 300));
    m.row_mut(0).assign(&l);
    m.row_mut(1).assign(&r);
    AudioSamples::new_multi_channel(m, sample_rate!(8000)).unwrap()
}

/// Largest absolute difference in LSB between two equal-length sample lists.
fn max_lsb_diff<T: StandardSample>(a: &[T], b: &[T]) -> f64 {
    assert_eq!(a.len(), b.len(), "length mismatch");
    a.iter()
        .zip(b)
        .map(|(&x, &y)| {
            let x: f64 = x.cast_into();
            let y: f64 = y.cast_into();
            (x - y).abs()
        })
        .fold(0.0, f64::max)
}

/// The `f64` result `r`, re-quantised to the sample type of `like`.
fn requantise<T>(r: &AudioSamples<'_, f64>, _like: &AudioSamples<'_, T>) -> Vec<T>
where
    T: StandardSample,
    f64: ConvertTo<T> + ConvertFrom<T>,
{
    r.to_format::<T>().as_vec()
}

/// Runs `$op` (with `$a` bound to the audio) on `$audio` and on its `f64`
/// conversion, and asserts that the integer result is within 1 LSB of the
/// `f64` result re-quantised to the integer type.
macro_rules! assert_close_to_f64_path {
    ($audio:expr, |$a:ident| $op:expr, $what:expr) => {{
        let audio = $audio;
        let int = {
            let $a = &audio;
            $op
        };
        let reference = {
            let $a = &audio.to_format::<f64>();
            $op
        };
        let d = max_lsb_diff(&int.as_vec(), &requantise(&reference, &int));
        assert!(
            d <= 1.0,
            "{} ({:?}): {d} LSB from the f64 path",
            $what,
            audio.as_vec().first()
        );
    }};
}

/// [`assert_close_to_f64_path`] for `i16`, `I24`, `i32` and `u8` versions of
/// the same `i16` audio.
macro_rules! assert_close_all_int_types {
    ($audio:expr, |$a:ident| $op:expr, $what:expr) => {{
        let base: AudioSamples<'static, i16> = $audio;
        assert_close_to_f64_path!(base.clone(), |$a| $op, concat!($what, " i16"));
        assert_close_to_f64_path!(base.to_format::<I24>(), |$a| $op, concat!($what, " I24"));
        assert_close_to_f64_path!(base.to_format::<i32>(), |$a| $op, concat!($what, " i32"));
        assert_close_to_f64_path!(base.to_format::<u8>(), |$a| $op, concat!($what, " u8"));
    }};
}

// ---------------------------------------------------------------------------
// apply_window_f64
// ---------------------------------------------------------------------------

#[test]
fn apply_window_f64_i16_fractional_coefficients() {
    let audio =
        AudioSamples::new_mono(array![16384i16, 16384, -16384, 1001], sample_rate!(8000)).unwrap();
    let w = [1.0, 0.5, 0.25, 0.3];
    let out: AudioSamples<'_, i16> = audio
        .apply_window_f64(NonEmptySlice::new(&w).unwrap())
        .unwrap();
    // 1001 · 0.3 = 300.3 -> 300; the others are exact.
    assert_eq!(out.as_vec(), vec![16384, 8192, -4096, 300]);
}

#[test]
fn apply_window_f64_matches_f64_path_for_integer_types() {
    let n = 256;
    let hann: Vec<f64> = (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / n as f64).cos())
        .collect();
    let w = NonEmptySlice::new(&hann).unwrap();
    let base = AudioSamples::new_mono(test_signal(n), sample_rate!(8000)).unwrap();
    assert_close_all_int_types!(base, |a| a.apply_window_f64(w).unwrap(), "apply_window_f64");
    let w = NonEmptySlice::new(&hann[..256]).unwrap();
    let st = stereo_i16();
    let st = AudioSamples::new_multi_channel(
        st.as_multi_channel()
            .unwrap()
            .slice(ndarray::s![.., ..256])
            .to_owned(),
        sample_rate!(8000),
    )
    .unwrap();
    assert_close_all_int_types!(
        st,
        |a| a.apply_window_f64(w).unwrap(),
        "apply_window_f64 stereo"
    );
}

#[test]
#[allow(deprecated)] // compares against the deprecated sample-typed window
fn apply_window_f64_equals_deprecated_window_for_floats() {
    let x = test_signal_f32(128);
    let w32: Vec<f32> = (0..128).map(|i| (i as f32 / 127.0).sqrt()).collect();
    let w64: Vec<f64> = w32.iter().map(|&w| f64::from(w)).collect();
    let audio = AudioSamples::new_mono(x, sample_rate!(8000)).unwrap();
    let typed = audio
        .apply_window(NonEmptySlice::new(&w32).unwrap())
        .unwrap();
    let wide = audio
        .apply_window_f64(NonEmptySlice::new(&w64).unwrap())
        .unwrap();
    assert_eq!(typed.as_vec(), wide.as_vec());
}

#[test]
fn apply_window_f64_rejects_wrong_length() {
    let audio = AudioSamples::new_mono(array![1i16, 2, 3], sample_rate!(8000)).unwrap();
    let err = audio
        .apply_window_f64(NonEmptySlice::new(&[1.0, 1.0]).unwrap())
        .unwrap_err();
    assert!(matches!(err, AudioSampleError::Layout(_)), "{err:?}");
}

// ---------------------------------------------------------------------------
// apply_filter_f64
// ---------------------------------------------------------------------------

fn lowpass_taps(m: usize) -> Vec<f64> {
    let c = (m - 1) as f64 / 2.0;
    (0..m)
        .map(|k| {
            let n = k as f64 - c;
            let s = if n == 0.0 {
                0.2
            } else {
                (0.2 * std::f64::consts::PI * n).sin() / (std::f64::consts::PI * n)
            };
            s * (0.54 - 0.46 * (2.0 * std::f64::consts::PI * k as f64 / (m - 1) as f64).cos())
        })
        .collect()
}

#[test]
fn apply_filter_f64_taps_above_one_on_i16() {
    let audio = AudioSamples::new_mono(test_signal(256), sample_rate!(8000)).unwrap();
    let h = [1.5, -0.5];
    let out = audio
        .apply_filter_f64(NonEmptySlice::new(&h).unwrap())
        .unwrap();
    let reference = audio
        .to_format::<f64>()
        .apply_filter_f64(NonEmptySlice::new(&h).unwrap())
        .unwrap()
        .to_format::<i16>();
    assert!(max_lsb_diff(&out.as_vec(), &reference.as_vec()) <= 1.0);
    // y[0] = 1.5·x[1] − 0.5·x[0], rounded.
    let x = audio.as_vec();
    let want = (1.5 * f64::from(x[1]) - 0.5 * f64::from(x[0])).round() as i16;
    assert!((i32::from(out[0]) - i32::from(want)).abs() <= 1);
}

#[test]
fn apply_filter_f64_matches_f64_path_for_integer_types() {
    for m in [31usize, 200] {
        // 200 taps takes the FFT path when `transforms` is enabled.
        let h = lowpass_taps(m);
        let h = NonEmptySlice::new(&h).unwrap();
        let base = AudioSamples::new_mono(test_signal(1024), sample_rate!(8000)).unwrap();
        assert_close_all_int_types!(base, |a| a.apply_filter_f64(h).unwrap(), "apply_filter_f64");
    }
}

#[test]
fn apply_filter_f64_equals_apply_filter_for_floats() {
    let audio = AudioSamples::new_mono(test_signal_f32(1024), sample_rate!(8000)).unwrap();
    for m in [7usize, 200] {
        let h32: Vec<f32> = lowpass_taps(m).iter().map(|&c| c as f32).collect();
        let h64: Vec<f64> = h32.iter().map(|&c| f64::from(c)).collect();
        let typed = audio
            .apply_filter(NonEmptySlice::new(&h32).unwrap())
            .unwrap();
        let wide = audio
            .apply_filter_f64(NonEmptySlice::new(&h64).unwrap())
            .unwrap();
        assert_eq!(typed.as_vec(), wide.as_vec(), "{m} taps");
    }
}

// ---------------------------------------------------------------------------
// normalize Mean / Median / ZScore, remove_dc_offset
// ---------------------------------------------------------------------------

#[test]
fn normalize_mean_rounds_integer_results() {
    let a = AudioSamples::new_mono(array![1i16, 2, 4], sample_rate!(8000)).unwrap();
    // mean 7/3: diffs -1.33, -0.33, 1.67 (truncation would give 1 for the last)
    assert_eq!(
        a.normalize(NormalizationConfig::mean()).unwrap().as_vec(),
        vec![-1, 0, 2]
    );
}

#[test]
fn normalize_median_rounds_integer_results() {
    let a = AudioSamples::new_mono(array![1i16, 2, 5, 8], sample_rate!(8000)).unwrap();
    // median 3.5: diffs -2.5, -1.5, 1.5, 4.5, rounded half away from zero
    // (truncation would give -2, -1, 1, 4)
    assert_eq!(
        a.normalize(NormalizationConfig::median()).unwrap().as_vec(),
        vec![-3, -2, 2, 5]
    );
}

#[test]
fn normalize_mean_and_dc_removal_match_f64_path() {
    let base = AudioSamples::new_mono(test_signal(512), sample_rate!(8000)).unwrap();
    // Mean removal acts on raw values; for signed types these are the f64
    // values times a constant per sign. u8 (offset binary) is not checked.
    assert_close_to_f64_path!(
        base.clone(),
        |a| a.normalize(NormalizationConfig::mean()).unwrap(),
        "normalize(mean) i16"
    );
    assert_close_to_f64_path!(
        base.clone(),
        |a| a.remove_dc_offset().unwrap(),
        "remove_dc_offset i16"
    );
    assert_close_to_f64_path!(
        base.to_format::<I24>(),
        |a| a.remove_dc_offset().unwrap(),
        "remove_dc_offset I24"
    );
    assert_close_to_f64_path!(
        base.to_format::<i32>(),
        |a| a.remove_dc_offset().unwrap(),
        "remove_dc_offset i32"
    );
}

#[test]
fn normalize_mean_saturates_instead_of_wrapping_i24() {
    let a =
        AudioSamples::new_mono(array![I24::MIN, I24::MAX, I24::MAX], sample_rate!(8000)).unwrap();
    let out = a.normalize(NormalizationConfig::mean()).unwrap().as_vec();
    // mean ≈ 2796202: MIN − mean is below I24::MIN and saturates.
    assert_eq!(out[0], I24::MIN);
}

#[test]
fn remove_dc_offset_rounds_integer_results() {
    let a = AudioSamples::new_mono(array![1i16, 2, 4], sample_rate!(8000)).unwrap();
    assert_eq!(a.remove_dc_offset().unwrap().as_vec(), vec![-1, 0, 2]);
}

#[test]
fn normalize_zscore_rejects_integer_types() {
    let a = AudioSamples::new_mono(test_signal(64), sample_rate!(8000)).unwrap();
    let err = a.normalize(NormalizationConfig::zscore()).unwrap_err();
    assert!(
        matches!(err, AudioSampleError::Unsupported { .. }),
        "{err:?}"
    );
    let err = a
        .to_format::<u8>()
        .normalize(NormalizationConfig::zscore())
        .unwrap_err();
    assert!(
        matches!(err, AudioSampleError::Unsupported { .. }),
        "{err:?}"
    );
    // Float input still works, and the f64 route is the documented workaround.
    let z = a
        .to_format::<f64>()
        .normalize(NormalizationConfig::zscore())
        .unwrap();
    let v = z.as_vec();
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    assert!(mean.abs() < 1e-12);
}

#[test]
fn normalize_float_results_are_exact_f64_formula() {
    let x = test_signal_f32(97);
    let a = AudioSamples::new_mono(x.clone(), sample_rate!(8000)).unwrap();
    let mean = x.iter().map(|&v| f64::from(v)).sum::<f64>() / x.len() as f64;
    let want: Vec<f32> = x.iter().map(|&v| (f64::from(v) - mean) as f32).collect();
    assert_eq!(
        a.normalize(NormalizationConfig::mean()).unwrap().as_vec(),
        want
    );
    assert_eq!(a.remove_dc_offset().unwrap().as_vec(), want);
}

// ---------------------------------------------------------------------------
// channel operations
// ---------------------------------------------------------------------------

#[test]
fn to_mono_average_center_and_weighted_round_to_nearest() {
    let st = AudioSamples::new_multi_channel(
        array![[1i16, 16384, -3, 1001], [2, 16384, -4, 1002]],
        sample_rate!(8000),
    )
    .unwrap();
    // (1+2)/2 = 1.5 -> 2, (-3-4)/2 = -3.5 -> -4 (truncation would give 1 and -3)
    let avg = st.to_mono(MonoConversionMethod::Average).unwrap().as_vec();
    assert_eq!(avg, vec![2, 16384, -4, 1002]);
    // Center of a stereo pair is the average.
    assert_eq!(
        st.to_mono(MonoConversionMethod::Center).unwrap().as_vec(),
        avg
    );
    // Weighted with [0.5, 0.5] is the average.
    let weighted = st
        .to_mono(MonoConversionMethod::Weighted(vec![0.5, 0.5]))
        .unwrap()
        .as_vec();
    assert!(
        max_lsb_diff(&weighted, &avg) <= 1.0,
        "{weighted:?} vs {avg:?}"
    );
    assert_eq!(weighted[1], 16384);
}

#[test]
fn to_mono_matches_f64_path() {
    let st = stereo_i16();
    let three = {
        let v = st.as_multi_channel().unwrap().to_owned();
        let mut m = Array2::<i16>::zeros((3, 300));
        m.row_mut(0).assign(&v.row(0));
        m.row_mut(1).assign(&v.row(1));
        m.row_mut(2).assign(&v.row(0).mapv(|x| x / 3));
        AudioSamples::new_multi_channel(m, sample_rate!(8000)).unwrap()
    };
    for method in [
        MonoConversionMethod::Average,
        MonoConversionMethod::Center,
        MonoConversionMethod::Weighted(vec![0.7, 0.2, 0.4]),
    ] {
        for audio in [&st, &three] {
            let m = match (&method, audio.num_channels().get()) {
                (MonoConversionMethod::Weighted(_), 2) => {
                    MonoConversionMethod::Weighted(vec![0.7, 0.2])
                }
                _ => method.clone(),
            };
            assert_close_to_f64_path!(
                audio.clone(),
                |a| a.to_mono(m.clone()).unwrap(),
                "to_mono i16"
            );
            assert_close_to_f64_path!(
                audio.to_format::<I24>(),
                |a| a.to_mono(m.clone()).unwrap(),
                "to_mono I24"
            );
            assert_close_to_f64_path!(
                audio.to_format::<i32>(),
                |a| a.to_mono(m.clone()).unwrap(),
                "to_mono i32"
            );
        }
    }
}

#[test]
fn pan_and_balance_round_and_saturate() {
    let st = AudioSamples::new_multi_channel(
        array![[1003i16, 32767, -32768], [1003, 20000, -32768]],
        sample_rate!(8000),
    )
    .unwrap();
    // pan(0.7): left gain 0.3, right gain 1.7.
    // 1003·0.3 = 300.9 -> 301 (not truncated to 300); 1003·1.7 = 1705.1 -> 1705;
    // 20000·1.7 and −32768·1.7 saturate.
    let p = st.pan(0.7).unwrap();
    assert_eq!(p.as_vec(), vec![301, 9830, -9830, 1705, 32767, -32768]);
    assert_eq!(st.balance(0.7).unwrap().as_vec(), p.as_vec());
}

#[test]
fn balance_i24_saturates_instead_of_wrapping() {
    let st = AudioSamples::new_multi_channel(
        array![[I24::MAX, I24::MAX], [I24::MAX, I24::MAX]],
        sample_rate!(8000),
    )
    .unwrap();
    // Right gain 2: saturates at I24::MAX rather than wrapping to I24::MIN.
    let out = st.balance(1.0).unwrap().as_vec();
    assert_eq!(
        out,
        vec![
            I24::try_from_i32(0).unwrap(),
            I24::try_from_i32(0).unwrap(),
            I24::MAX,
            I24::MAX
        ]
    );
}

#[test]
fn balance_u8_scales_around_mid_scale() {
    let st = AudioSamples::new_multi_channel(array![[200u8, 128], [200, 60]], sample_rate!(8000))
        .unwrap();
    // Left gain 0.5: 200 is +72 from 128 -> +36 = 164; 128 stays 128.
    // Right gain 1.5: +72 -> +108 = 236; 60 is −68 -> −102 = 26.
    // (Scaling the raw codes would give 100, 64, 255, 90.)
    assert_eq!(st.balance(0.5).unwrap().as_vec(), vec![164, 128, 236, 26]);
}

#[test]
fn pan_matches_f64_path() {
    let st = stereo_i16();
    for p in [-0.9, -0.3, 0.0, 0.45, 1.0] {
        assert_close_all_int_types!(st.clone(), |a| a.pan(p).unwrap(), "pan");
        assert_close_all_int_types!(st.clone(), |a| a.balance(p).unwrap(), "balance");
    }
}

#[test]
fn to_stereo_pan_rounds() {
    let m = AudioSamples::new_mono(array![7i16, 10000, -10000], sample_rate!(8000)).unwrap();
    // gain cos(π/4) = 0.70711: 7 -> 4.95 -> 5 (not truncated to 4)
    let out = m
        .to_stereo(StereoConversionMethod::Pan(0.0))
        .unwrap()
        .as_vec();
    assert_eq!(out, vec![5, 7071, -7071, 5, 7071, -7071]);
    let base = AudioSamples::new_mono(test_signal(200), sample_rate!(8000)).unwrap();
    assert_close_all_int_types!(
        base,
        |a| a.to_stereo(StereoConversionMethod::Pan(0.3)).unwrap(),
        "to_stereo(Pan(0.3))"
    );
}

#[test]
fn channel_ops_float_results_are_exact_f64_formula() {
    let x = test_signal_f32(64);
    let y = test_signal_f32(65).slice(ndarray::s![1..]).to_owned();
    let mut m = Array2::<f32>::zeros((2, 64));
    m.row_mut(0).assign(&x);
    m.row_mut(1).assign(&y);
    let st = AudioSamples::new_multi_channel(m, sample_rate!(8000)).unwrap();
    // pan: raw f64 product cast to f32
    let want: Vec<f32> = x
        .iter()
        .map(|&v| (f64::from(v) * 0.6) as f32)
        .chain(y.iter().map(|&v| (f64::from(v) * 1.4) as f32))
        .collect();
    assert_eq!(st.pan(0.4).unwrap().as_vec(), want);
    // weighted: f64 sum cast to f32
    let want: Vec<f32> = x
        .iter()
        .zip(y.iter())
        .map(|(&l, &r)| (f64::from(l) * 0.25 + f64::from(r) * 0.75) as f32)
        .collect();
    assert_eq!(
        st.to_mono(MonoConversionMethod::Weighted(vec![0.25, 0.75]))
            .unwrap()
            .as_vec(),
        want
    );
    // center (stereo): f64 average cast to f32
    let want: Vec<f32> = x
        .iter()
        .zip(y.iter())
        .map(|(&l, &r)| ((f64::from(l) + f64::from(r)) / 2.0) as f32)
        .collect();
    assert_eq!(
        st.to_mono(MonoConversionMethod::Center).unwrap().as_vec(),
        want
    );
}
