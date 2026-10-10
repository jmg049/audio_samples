//! Inputs of one benchmark configuration.

use std::cell::OnceCell;
use std::hint::black_box;
use std::marker::PhantomData;

use audio_samples::operations::spectrum::StftConfig;
use std::num::NonZeroU32 as SampleRate;

use audio_samples::{AudioFloat, AudioSamples, AudioSpectrum};
use ndarray::{Array1, Array2, Array3};
use num_complex::Complex;

/// One timed call. Builders compute everything that is not part of the call
/// (configs, prepared inputs) before creating it and move it in.
pub type Thunk<'a> = Box<dyn FnMut() + 'a>;

/// Wraps a closure whose result is passed through `black_box` and dropped
/// inside the timed region (librosa's result is freed inside it too).
pub fn thunk<'a, R>(mut f: impl FnMut() -> R + 'a) -> Option<Thunk<'a>> {
    Some(Box::new(move || {
        black_box(f());
    }))
}

/// An audio configuration: `{sr}_{seconds}s_{mono|stereo}_{f32|f64}`.
pub struct AudioCtx<F: AudioFloat> {
    pub sr: u32,
    pub seconds: u32,
    pub channels: usize,
    /// The signal (same samples as the Python half).
    pub audio: AudioSamples<'static, F>,
    /// The signal as `(channels, n)`.
    pub y: Array2<F>,
    d: OnceCell<Array3<Complex<F>>>,
}

impl<F: AudioFloat> AudioCtx<F> {
    pub fn load(config: &str, signals: &str) -> Self {
        let parts: Vec<&str> = config.split('_').collect();
        let sr: u32 = parts[0].parse().expect("rate");
        let seconds: u32 = parts[1].trim_end_matches('s').parse().expect("seconds");
        let channels = if parts[2] == "mono" { 1 } else { 2 };
        let path = format!("{signals}/sig_{sr}_{seconds}s_{channels}ch.f64");
        let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
        let v: Vec<F> = bytes
            .chunks_exact(8)
            .map(|b| F::of(f64::from_le_bytes(b.try_into().unwrap())))
            .collect();
        let n = v.len() / channels;
        let y = Array2::from_shape_vec((channels, n), v).unwrap();
        let rate = SampleRate::new(sr).unwrap();
        let audio = if channels == 1 {
            AudioSamples::new_mono(y.row(0).to_owned(), rate).unwrap()
        } else {
            AudioSamples::new_multi_channel(y.clone(), rate).unwrap()
        };
        Self {
            sr,
            seconds,
            channels,
            audio,
            y,
            d: OnceCell::new(),
        }
    }

    pub fn rate(&self) -> SampleRate {
        SampleRate::new(self.sr).unwrap()
    }

    /// Samples per channel.
    pub fn n(&self) -> usize {
        self.y.ncols()
    }

    /// The first channel as a 1-D array.
    pub fn mono(&self) -> Array1<F> {
        self.y.row(0).to_owned()
    }

    /// `librosa.stft(y)` with librosa's defaults, `(channels, 1025, frames)`.
    pub fn stft(&self) -> &Array3<Complex<F>> {
        self.d
            .get_or_init(|| self.audio.stft_matrix(&StftConfig::librosa()).unwrap())
    }

    /// `|stft|`.
    pub fn s_mag(&self) -> Array3<F> {
        self.stft().mapv(|c| c.norm())
    }

    /// `|stft|^2`.
    pub fn s_pow(&self) -> Array3<F> {
        self.stft().mapv(|c| c.norm_sqr())
    }
}

/// No input (filterbanks, windows, frequency tables). `F` is the requested
/// output type where the function has one.
pub struct ConstCtx<F: AudioFloat>(PhantomData<F>);

impl<F: AudioFloat> ConstCtx<F> {
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

/// Length of the 1-D arrays of the `vec` configurations.
pub const VEC_N: usize = 100_000;

/// A `vec` configuration: unit conversions on 1-D arrays.
pub struct VecCtx<F: AudioFloat>(PhantomData<F>);

impl<F: AudioFloat> VecCtx<F> {
    pub fn new() -> Self {
        Self(PhantomData)
    }

    /// `VEC_N` values over `[lo, hi]`, log-spaced when `lo > 0`
    /// (numpy `geomspace` / `linspace`, as the Python half).
    pub fn values(&self, lo: f64, hi: f64) -> Array1<F> {
        let n = VEC_N;
        Array1::from_shape_fn(n, |i| {
            let t = i as f64 / (n - 1) as f64;
            let v = if lo > 0.0 {
                (lo.ln() + t * (hi.ln() - lo.ln())).exp()
            } else {
                lo + t * (hi - lo)
            };
            F::of(v)
        })
    }
}

/// A `seq` configuration: fixed-size matrices for sequence alignment and
/// decoding.
pub struct SeqCtx<F: AudioFloat>(PhantomData<F>);

impl<F: AudioFloat> SeqCtx<F> {
    pub fn new() -> Self {
        Self(PhantomData)
    }

    /// Deterministic uniform `[0, 1)` matrix (64-bit LCG). The Python half
    /// uses numpy's generator: same shape and distribution, different values.
    pub fn matrix(&self, rows: usize, cols: usize, seed: u64) -> Array2<F> {
        let mut s = seed;
        Array2::from_shape_fn((rows, cols), |_| {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            F::of((s >> 11) as f64 / (1u64 << 53) as f64)
        })
    }
}
