//! The floating-point types used for analysis output.
//!
//! Analysis operations (spectrograms, mel spectrograms, MFCCs, chroma, CQT,
//! onset envelopes, spectral features, pitch tracks) compute in a floating-point
//! type chosen from the input sample type: the smallest float type that
//! represents every input sample exactly.
//!
//! | Sample type `T` | `T::Float` |
//! |---|---|
//! | `u8`, `i16`, [`I24`](crate::I24), `f32` | `f32` |
//! | `i32`, `f64` | `f64` |
//!
//! [`AudioSample::Float`] names that type, and [`AudioFloat`] is the trait that
//! both float types implement. `AudioFloat` collects every capability the
//! analysis code needs (`num_traits::Float`, `ndarray` scalar operations,
//! `Sum`, FFT support, ...), so generic code needs the single bound
//! `F: AudioFloat` and no further `num_traits` bounds.
//!
//! Computing in `f32` halves the memory of every intermediate array and of the
//! output. Audio of type `i32` or `f64` is analysed in `f64`, because `f32`
//! cannot hold every `i32` sample value exactly.
//!
//! ## Example
//!
//! ```
//! use audio_samples::{AudioFloat, AudioSample};
//!
//! // The analysis float type of 16-bit audio is f32.
//! fn float_label<T: AudioSample>() -> &'static str {
//!     <T::Float as AudioSample>::LABEL
//! }
//! assert_eq!(float_label::<i16>(), "f32");
//! assert_eq!(float_label::<i32>(), "f64");
//!
//! // Generic float code reads like ordinary float code.
//! fn rms<F: AudioFloat>(x: &[F]) -> F {
//!     let sum: F = x.iter().map(|&v| v * v).sum();
//!     (sum / F::of_usize(x.len())).sqrt()
//! }
//! assert!((rms(&[3.0_f32, 4.0]) - 12.5_f32.sqrt()).abs() < 1e-6);
//! ```

use std::any::{Any, TypeId};
use std::iter::{Product, Sum};

use ndarray::{Array2, ArrayView2, Axis, CowArray, Ix2, NdFloat};
use num_traits::{FloatConst, NumAssign};

use crate::repr::SampleRate;
use crate::{
    AudioSample, AudioSampleError, AudioSampleResult, AudioSamples, ParameterError, StandardSample,
};

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

/// FFT support for [`AudioFloat`] when the `transforms` feature is enabled.
#[cfg(feature = "transforms")]
#[doc(hidden)]
pub trait MaybeFftNum: rustfft::FftNum {}
#[cfg(feature = "transforms")]
impl<T: rustfft::FftNum> MaybeFftNum for T {}

/// FFT support for [`AudioFloat`] when the `transforms` feature is enabled.
#[cfg(not(feature = "transforms"))]
#[doc(hidden)]
pub trait MaybeFftNum {}
#[cfg(not(feature = "transforms"))]
impl<T> MaybeFftNum for T {}

/// A floating-point type that analysis code computes in: `f32` or `f64`.
///
/// # Purpose
///
/// `AudioFloat` is the one bound generic analysis code needs. It includes
/// `num_traits::Float` and `FloatConst` (through `ndarray::NdFloat`), the
/// compound assignment operators, `Sum` and `Product`, `ndarray` scalar and
/// matrix-product support, `Send`, `Sync`, `Debug` and `Display`, and (with the
/// `transforms` feature) `rustfft::FftNum`. It is also a [`StandardSample`], so
/// audio of type `f32` or `f64` can be built directly from analysis output.
///
/// The trait is sealed: only `f32` and `f64` implement it.
///
/// # Constants and conversions
///
/// Write literals and conversions with [`of`](Self::of) and
/// [`as_f64`](Self::as_f64): `F::of(0.5) * x`. Constants derived from `f64`
/// (filter tables, window coefficients) can be computed in `f64` once and cast
/// with `F::of`.
///
/// Note that `F::MAX` and `F::MIN` are the [`AudioSample`] full-scale constants
/// (`1.0` and `-1.0`), not the largest finite values. Use
/// `F::max_value()` / `F::min_value()` for those.
pub trait AudioFloat:
    sealed::Sealed
    + StandardSample
    + AudioSample<Float = Self>
    + NdFloat
    + FloatConst
    + NumAssign
    + Sum
    + Product
    + for<'a> Sum<&'a Self>
    + for<'a> Product<&'a Self>
    + MaybeFftNum
{
    /// Converts an `f64` to this type (`v as f32` for `f32`, identity for `f64`).
    fn of(v: f64) -> Self;

    /// Converts a `usize` to this type (`n as f32` or `n as f64`).
    fn of_usize(n: usize) -> Self;

    /// Converts this value to `f64` (exact for both types).
    fn as_f64(self) -> f64;

    /// Runs `f` with this thread's real-FFT planner for this type.
    ///
    /// The planner caches plans by length, so repeated transforms of one size
    /// reuse their twiddle factors.
    #[cfg(feature = "transforms")]
    #[doc(hidden)]
    fn with_real_fft_planner<R>(f: impl FnOnce(&mut realfft::RealFftPlanner<Self>) -> R) -> R;

    /// Runs `f` with this thread's complex-FFT planner for this type.
    #[cfg(feature = "transforms")]
    #[doc(hidden)]
    fn with_fft_planner<R>(f: impl FnOnce(&mut rustfft::FftPlanner<Self>) -> R) -> R;
}

#[cfg(feature = "transforms")]
thread_local! {
    static REAL_PLANNER_F32: std::cell::RefCell<realfft::RealFftPlanner<f32>> =
        std::cell::RefCell::new(realfft::RealFftPlanner::new());
    static REAL_PLANNER_F64: std::cell::RefCell<realfft::RealFftPlanner<f64>> =
        std::cell::RefCell::new(realfft::RealFftPlanner::new());
    static PLANNER_F32: std::cell::RefCell<rustfft::FftPlanner<f32>> =
        std::cell::RefCell::new(rustfft::FftPlanner::new());
    static PLANNER_F64: std::cell::RefCell<rustfft::FftPlanner<f64>> =
        std::cell::RefCell::new(rustfft::FftPlanner::new());
}

impl AudioFloat for f32 {
    #[inline]
    fn of(v: f64) -> Self {
        v as Self
    }

    #[inline]
    fn of_usize(n: usize) -> Self {
        n as Self
    }

    #[inline]
    fn as_f64(self) -> f64 {
        f64::from(self)
    }

    #[cfg(feature = "transforms")]
    fn with_real_fft_planner<R>(f: impl FnOnce(&mut realfft::RealFftPlanner<Self>) -> R) -> R {
        REAL_PLANNER_F32.with(|p| f(&mut p.borrow_mut()))
    }

    #[cfg(feature = "transforms")]
    fn with_fft_planner<R>(f: impl FnOnce(&mut rustfft::FftPlanner<Self>) -> R) -> R {
        PLANNER_F32.with(|p| f(&mut p.borrow_mut()))
    }
}

impl AudioFloat for f64 {
    #[inline]
    fn of(v: f64) -> Self {
        v
    }

    #[inline]
    fn of_usize(n: usize) -> Self {
        n as Self
    }

    #[inline]
    fn as_f64(self) -> f64 {
        self
    }

    #[cfg(feature = "transforms")]
    fn with_real_fft_planner<R>(f: impl FnOnce(&mut realfft::RealFftPlanner<Self>) -> R) -> R {
        REAL_PLANNER_F64.with(|p| f(&mut p.borrow_mut()))
    }

    #[cfg(feature = "transforms")]
    fn with_fft_planner<R>(f: impl FnOnce(&mut rustfft::FftPlanner<Self>) -> R) -> R {
        PLANNER_F64.with(|p| f(&mut p.borrow_mut()))
    }
}

/// The audio as a `(channels, samples)` view of its own sample type (a mono
/// signal is one row).
fn audio_rows_view<'a, T: StandardSample>(audio: &'a AudioSamples<'_, T>) -> ArrayView2<'a, T> {
    if let Some(m) = audio.as_mono() {
        m.as_view().insert_axis(Axis(0))
    } else if let Some(m) = audio.as_multi_channel() {
        m.as_view()
    } else {
        unreachable!("audio is either mono or multi-channel")
    }
}

/// `view` as a view of `F` when `T` and `F` are the same type, else `None`.
fn same_type_view<'a, T: 'static, F: 'static>(
    view: ArrayView2<'a, T>,
) -> Option<ArrayView2<'a, F>> {
    if TypeId::of::<T>() == TypeId::of::<F>() {
        // SAFETY: `T` and `F` are the same type, so the two view types are
        // identical; views are `Copy` and own nothing.
        Some(unsafe { std::mem::transmute_copy::<ArrayView2<'a, T>, ArrayView2<'a, F>>(&view) })
    } else {
        None
    }
}

/// `rows` as an `Array2<T>` without copying when `F` and `T` are the same
/// type, else `Err(rows)`.
fn same_type_array<F: 'static, T: 'static>(rows: Array2<F>) -> Result<Array2<T>, Array2<F>> {
    if TypeId::of::<T>() == TypeId::of::<F>() {
        let any: Box<dyn Any> = Box::new(rows);
        Ok(*any.downcast::<Array2<T>>().expect("the type ids are equal"))
    } else {
        Err(rows)
    }
}

/// Whether every element of `x` is finite. Contiguous data is checked in
/// fixed-size blocks without branching inside a block, so the check
/// vectorises.
fn all_finite<F: AudioFloat>(x: ArrayView2<'_, F>) -> bool {
    match x.as_slice_memory_order() {
        Some(s) => slice_finite(s),
        None => x.iter().all(|v| v.is_finite()),
    }
}

/// Whether every value of `s` is finite, compiled for AVX2 where the CPU has
/// it.
fn slice_finite<F: AudioFloat>(s: &[F]) -> bool {
    #[inline(always)]
    fn body<F: AudioFloat>(s: &[F]) -> bool {
        s.chunks(512)
            .all(|c| c.iter().fold(true, |ok, v| ok & v.is_finite()))
    }
    /// # Safety
    /// The CPU must support AVX2.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn avx2<F: AudioFloat>(s: &[F]) -> bool {
        body(s)
    }
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: the CPU supports AVX2 (checked just above).
        return unsafe { avx2(s) };
    }
    body(s)
}

/// The audio as a `(channels, samples)` array of its analysis float type,
/// converted with the crate's audio-aware conversion (no validation).
#[allow(dead_code)] // used by the analysis modules as they move to `T::Float`
pub(crate) fn audio_float_rows<T: StandardSample>(audio: &AudioSamples<'_, T>) -> Array2<T::Float> {
    audio_rows_view(audio).mapv(T::to_float)
}

/// The audio as `(channels, samples)` rows of its analysis float type,
/// rejecting NaN and infinite samples. Integer sample types cannot hold
/// non-finite values.
///
/// Audio of type `f32` or `f64` (`T == T::Float`) is borrowed, not copied;
/// other sample types are converted with the crate's audio-aware conversion.
///
/// # Errors
/// [`ParameterError`] if any sample is NaN or infinite.
#[allow(dead_code)] // used by the analysis modules as they move to `T::Float`
pub(crate) fn finite_float_rows<'a, T: StandardSample>(
    audio: &'a AudioSamples<'_, T>,
) -> AudioSampleResult<CowArray<'a, T::Float, Ix2>> {
    let view = audio_rows_view(audio);
    let Some(rows) = same_type_view::<T, T::Float>(view) else {
        // Integer samples are always finite.
        return Ok(CowArray::from(view.mapv(T::to_float)));
    };
    if !all_finite(rows) {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "audio",
            "audio buffer is not finite everywhere (contains NaN or infinity)",
        )));
    }
    Ok(CowArray::from(rows))
}

/// Builds audio of sample type `T` from `(channels, samples)` float rows,
/// converting every value with the crate's audio-aware conversion (integer
/// types round and saturate). When `T` is the float type itself the rows are
/// moved into the audio without a copy.
///
/// # Errors
/// An empty-data error naming `operation` if `rows` has no samples or no
/// channels.
#[allow(dead_code)] // used by the analysis modules as they move to `T::Float`
pub(crate) fn float_rows_to_audio<T: StandardSample>(
    rows: Array2<T::Float>,
    sample_rate: SampleRate,
    operation: &str,
) -> AudioSampleResult<AudioSamples<'static, T>> {
    if rows.is_empty() {
        return Err(AudioSampleError::empty_data(operation));
    }
    let converted = match same_type_array::<T::Float, T>(rows) {
        Ok(same) => same,
        Err(rows) => rows.mapv(T::from_float),
    };
    if converted.nrows() == 1 {
        AudioSamples::new_mono(converted.index_axis_move(Axis(0), 0), sample_rate)
    } else {
        AudioSamples::new_multi_channel(converted, sample_rate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{I24, sample_rate};
    use ndarray::array;

    fn label<T: AudioSample>() -> &'static str {
        <T::Float as AudioSample>::LABEL
    }

    #[test]
    fn float_type_is_smallest_exact_float() {
        assert_eq!(label::<u8>(), "f32");
        assert_eq!(label::<i16>(), "f32");
        assert_eq!(label::<I24>(), "f32");
        assert_eq!(label::<f32>(), "f32");
        assert_eq!(label::<i32>(), "f64");
        assert_eq!(label::<f64>(), "f64");
    }

    #[test]
    fn i24_samples_are_exact_in_f32() {
        // Every 24-bit integer is exactly representable in f32's 24-bit significand.
        for v in [I24::MAX.to_i32(), I24::MIN.to_i32(), 8_388_605, -3] {
            assert_eq!(v as f32 as i32, v);
        }
    }

    #[test]
    fn rows_round_trip_through_float() {
        let audio = AudioSamples::new_multi_channel(
            array![[1000_i16, -32768, 32767], [0, 5, -5]],
            sample_rate!(8000),
        )
        .unwrap();
        let rows: Array2<f32> = finite_float_rows(&audio).unwrap().into_owned();
        let back: AudioSamples<'static, i16> =
            float_rows_to_audio(rows, audio.sample_rate(), "test").unwrap();
        assert_eq!(back, audio);
    }

    #[test]
    fn to_float_matches_to_format_for_every_type() {
        use crate::AudioTypeConversion;
        let all_i16: Vec<i16> = (i16::MIN..=i16::MAX).collect();
        let audio =
            AudioSamples::new_mono(ndarray::Array1::from(all_i16), sample_rate!(8000)).unwrap();
        let ours = audio_float_rows(&audio);
        let reference = audio.to_format::<f32>();
        assert_eq!(ours.row(0), reference.as_mono().unwrap().as_view());
        let i32_audio = AudioSamples::new_mono(
            array![i32::MIN, -1, 0, 1, 123_456_789, i32::MAX],
            sample_rate!(8000),
        )
        .unwrap();
        let ours = audio_float_rows(&i32_audio);
        let reference = i32_audio.to_format::<f64>();
        assert_eq!(ours.row(0), reference.as_mono().unwrap().as_view());
    }

    #[test]
    fn non_finite_float_audio_is_rejected() {
        let audio = AudioSamples::new_mono(array![0.0_f32, f32::NAN], sample_rate!(8000)).unwrap();
        assert!(finite_float_rows(&audio).is_err());
    }
}
