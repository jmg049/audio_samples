//! Window functions and mel filterbanks with librosa / SciPy semantics.
//!
//! ## What
//!
//! This module holds the building blocks of the librosa-compatible spectral
//! pipeline in [`crate::operations::spectrum`]:
//!
//! - [`Window`] and [`get_window`]: analysis windows computed exactly as
//!   `scipy.signal.get_window`, in either the *periodic* form (`fftbins=True`,
//!   librosa's default, the right choice for spectral analysis) or the
//!   *symmetric* form (`fftbins=False`, used for filter design).
//! - [`window_sumsquare`] and [`window_bandwidth`]: the squared-window overlap
//!   envelope used to normalise an inverse STFT, and the equivalent noise
//!   bandwidth of a window (`librosa.filters.window_sumsquare` /
//!   `librosa.filters.window_bandwidth`).
//! - [`mel_filterbank`] with [`MelFilterbankConfig`]: the triangular mel
//!   filterbank of `librosa.filters.mel`, on either the Slaney or the HTK mel
//!   scale ([`MelScale`]) and with any of librosa's normalisations
//!   ([`MelFilterNorm`]).
//!
//! ## Why
//!
//! The spectrogram helpers on [`AudioTransforms`](crate::AudioTransforms) use
//! the `spectrograms` crate, whose windows are symmetric and whose mel
//! filterbank is fixed to the Slaney scale and is not exposed. librosa (and
//! therefore most published feature pipelines) uses periodic windows, a
//! Slaney-normalised filterbank and optionally the HTK scale. These functions
//! reproduce those conventions so that results match librosa to round-off.
//!
//! ## How
//!
//! Every function is a pure free function. All defaults follow librosa 1.0.
//!
//! ```rust
//! use audio_samples::nzu;
//! use audio_samples::operations::filterbanks::{
//!     MelFilterbankConfig, Window, get_window, mel_filterbank,
//! };
//!
//! // A periodic Hann window, as used by librosa.stft.
//! let w = get_window(&Window::Hann, 8, true).unwrap();
//! assert_eq!(w[0], 0.0);
//! assert!((w[4] - 1.0).abs() < 1e-15);
//!
//! // librosa.filters.mel(sr=22050, n_fft=2048): 128 Slaney bands, Slaney norm.
//! let mel = mel_filterbank(22050.0, nzu!(2048), &MelFilterbankConfig::default()).unwrap();
//! assert_eq!(mel.dim(), (128, 1025));
//! ```

use std::num::NonZeroUsize;

use ndarray::{Array1, Array2};

pub use crate::utils::audio_math::MelScale;
use crate::utils::audio_math::mel_frequencies_with_scale;
use crate::utils::{MAX_SIGNAL_ELEMENTS, MAX_TABLE_ELEMENTS, check_derived_len};
use crate::{AudioSampleError, AudioSampleResult, ParameterError};

// =============================================================================
// Windows
// =============================================================================

/// An analysis window, computed as `scipy.signal.get_window` computes it.
///
/// ## Purpose
///
/// Selects the taper applied to each frame before the FFT (STFT, ISTFT and
/// everything built on them). Each named variant corresponds to a SciPy window
/// name; parameterised windows carry their parameter.
///
/// ## Intended usage
///
/// Pass a `Window` to [`get_window`] (or put it in an
/// [`StftConfig`](crate::operations::spectrum::StftConfig)). The default is
/// [`Window::Hann`], librosa's default.
///
/// ## Invariants
///
/// None beyond the documented parameter ranges, which [`get_window`] checks:
/// `Kaiser::beta` must be finite, `Gaussian::std` finite and positive, and a
/// `Custom` window must have exactly the requested length and finite values.
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::filterbanks::{Window, get_window};
///
/// let kaiser = get_window(&Window::Kaiser { beta: 8.6 }, 16, true).unwrap();
/// assert_eq!(kaiser.len(), 16);
/// let custom = get_window(&Window::Custom(vec![0.5, 1.0, 0.5]), 3, true).unwrap();
/// assert_eq!(custom.to_vec(), vec![0.5, 1.0, 0.5]);
/// ```
#[derive(Debug, Clone, PartialEq, Default)]
pub enum Window {
    /// Rectangular window (all ones). SciPy `"boxcar"`.
    Rectangular,
    /// Hann (raised cosine) window, `0.5 - 0.5 cos`. SciPy `"hann"`; librosa's default.
    #[default]
    Hann,
    /// Hamming window, `0.54 - 0.46 cos`. SciPy `"hamming"`.
    Hamming,
    /// Classic Blackman window (coefficients 0.42, 0.5, 0.08). SciPy `"blackman"`.
    Blackman,
    /// 4-term Blackman-Harris window. SciPy `"blackmanharris"`.
    BlackmanHarris,
    /// Bartlett window: triangular with zero end points. SciPy `"bartlett"`.
    Bartlett,
    /// Triangular window with non-zero end points. SciPy `"triang"` (differs from
    /// [`Window::Bartlett`]).
    Triangular,
    /// Kaiser window with shape parameter `beta` (finite; larger means a wider main
    /// lobe and lower side lobes, e.g. 8.6 is close to Blackman). SciPy `("kaiser", beta)`.
    Kaiser {
        /// Shape parameter. Must be finite.
        beta: f64,
    },
    /// Gaussian window with standard deviation `std` in samples (finite, > 0).
    /// SciPy `("gaussian", std)`.
    Gaussian {
        /// Standard deviation in samples. Must be finite and positive.
        std: f64,
    },
    /// Explicit window coefficients. The length must equal the requested window
    /// length; the `periodic` flag of [`get_window`] does not apply (as in librosa,
    /// where an array window is used as given).
    Custom(Vec<f64>),
}

/// Coefficients of the generalised cosine windows (SciPy `general_cosine`).
fn cosine_coefficients(window: &Window) -> Option<&'static [f64]> {
    match window {
        Window::Hann => Some(&[0.5, 0.5]),
        Window::Hamming => Some(&[0.54, 0.46]),
        Window::Blackman => Some(&[0.42, 0.50, 0.08]),
        Window::BlackmanHarris => Some(&[0.35875, 0.48829, 0.14128, 0.01168]),
        _ => None,
    }
}

/// Computes an analysis window of length `n`, exactly as `scipy.signal.get_window`.
///
/// This is librosa's `librosa.filters.get_window(window, n, fftbins=periodic)`.
///
/// # Arguments
///
/// * `window` - The window type and its parameters.
/// * `n` - Window length in samples. `0` gives an empty array and `1` gives `[1.0]`
///   (as SciPy does) for every named window.
/// * `periodic` - `true` for the periodic ("DFT-even", SciPy `fftbins=True`) window
///   used for spectral analysis, which is librosa's default; `false` for the
///   symmetric window used for filter design. A periodic window of length `n` is the
///   symmetric window of length `n + 1` with its last sample dropped. Ignored for
///   [`Window::Custom`].
///
/// # Returns
///
/// The `n` window coefficients.
///
/// # Errors
///
/// [`ParameterError`] if a window parameter is out of range (non-finite Kaiser
/// `beta`, non-positive Gaussian `std`), if a [`Window::Custom`] has a length
/// other than `n` or a non-finite value, or if `n` exceeds 2^31 samples.
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::filterbanks::{Window, get_window};
///
/// let periodic = get_window(&Window::Hann, 4, true).unwrap();
/// assert!((periodic[1] - 0.5).abs() < 1e-15);   // 0.5 - 0.5 cos(2 pi / 4)
/// let symmetric = get_window(&Window::Hann, 4, false).unwrap();
/// assert!((symmetric[1] - 0.75).abs() < 1e-15); // 0.5 - 0.5 cos(2 pi / 3)
/// ```
///
/// # References
///
/// F. J. Harris, "On the use of windows for harmonic analysis with the discrete
/// Fourier transform," *Proc. IEEE* 66(1), 1978; SciPy `scipy.signal.get_window`.
pub fn get_window(window: &Window, n: usize, periodic: bool) -> AudioSampleResult<Array1<f64>> {
    validate_window(window)?;
    check_derived_len("n", "a window", n as f64, MAX_SIGNAL_ELEMENTS)?;
    if let Window::Custom(coeffs) = window {
        if coeffs.len() != n {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "window",
                format!(
                    "custom window has {} coefficients but a window of length {n} was requested",
                    coeffs.len()
                ),
            )));
        }
        return Ok(Array1::from(coeffs.clone()));
    }
    if n <= 1 {
        return Ok(Array1::ones(n));
    }
    // SciPy: a periodic window is a symmetric window of length n + 1, truncated.
    let m = if periodic { n + 1 } else { n };
    let mut w = symmetric_window(window, m);
    w.truncate(n);
    Ok(Array1::from(w))
}

/// Checks the parameters of a window (lengths are checked by the callers).
fn validate_window(window: &Window) -> AudioSampleResult<()> {
    match window {
        Window::Kaiser { beta } if !beta.is_finite() => Err(AudioSampleError::Parameter(
            ParameterError::invalid_value("beta", "Kaiser beta must be finite"),
        )),
        Window::Gaussian { std } if !(std.is_finite() && *std > 0.0) => {
            Err(AudioSampleError::Parameter(ParameterError::out_of_range(
                "std",
                std,
                "> 0",
                "finite",
                "Gaussian window standard deviation must be finite and positive",
            )))
        }
        Window::Custom(c) if c.iter().any(|x| !x.is_finite()) => Err(AudioSampleError::Parameter(
            ParameterError::invalid_value("window", "custom window contains non-finite values"),
        )),
        _ => Ok(()),
    }
}

/// The symmetric window of length `m >= 2` for a named (non-custom) window,
/// following SciPy's formulas operation by operation.
fn symmetric_window(window: &Window, m: usize) -> Vec<f64> {
    let mf = m as f64;
    if let Some(a) = cosine_coefficients(window) {
        // fac = numpy.linspace(-pi, pi, m); w = sum_k a[k] cos(k * fac)
        let step = (std::f64::consts::PI - (-std::f64::consts::PI)) / (mf - 1.0);
        return (0..m)
            .map(|i| {
                let fac = if i == m - 1 {
                    std::f64::consts::PI
                } else {
                    i as f64 * step + (-std::f64::consts::PI)
                };
                let mut w = 0.0;
                for (k, &ak) in a.iter().enumerate() {
                    w += ak * (k as f64 * fac).cos();
                }
                w
            })
            .collect();
    }
    match window {
        Window::Bartlett => (0..m)
            .map(|i| {
                let n = i as f64;
                if n <= (mf - 1.0) / 2.0 {
                    2.0 * n / (mf - 1.0)
                } else {
                    2.0 - 2.0 * n / (mf - 1.0)
                }
            })
            .collect(),
        Window::Triangular => {
            let half: Vec<f64> = (1..=m.div_ceil(2))
                .map(|k| {
                    let n = k as f64;
                    if m.is_multiple_of(2) {
                        (2.0 * n - 1.0) / mf
                    } else {
                        2.0 * n / (mf + 1.0)
                    }
                })
                .collect();
            let mut w = half.clone();
            let mirrored = if m.is_multiple_of(2) {
                &half[..]
            } else {
                &half[..half.len() - 1]
            };
            w.extend(mirrored.iter().rev());
            w
        }
        Window::Kaiser { beta } => {
            let alpha = (mf - 1.0) / 2.0;
            let denom = bessel_i0(*beta);
            (0..m)
                .map(|i| {
                    let r = (i as f64 - alpha) / alpha;
                    bessel_i0(beta * (1.0 - r * r).sqrt()) / denom
                })
                .collect()
        }
        Window::Gaussian { std } => {
            let sig2 = 2.0 * std * std;
            (0..m)
                .map(|i| {
                    let n = i as f64 - (mf - 1.0) / 2.0;
                    (-(n * n) / sig2).exp()
                })
                .collect()
        }
        // Cosine windows were handled above and custom windows never reach here,
        // so this arm is the rectangular window.
        Window::Rectangular
        | Window::Hann
        | Window::Hamming
        | Window::Blackman
        | Window::BlackmanHarris
        | Window::Custom(_) => vec![1.0; m],
    }
}

/// Chebyshev coefficients for I0 on `[0, 8]` (Cephes `i0`, as used by NumPy/SciPy).
#[allow(clippy::excessive_precision)] // Cephes coefficients, kept verbatim
const I0_A: [f64; 30] = [
    -4.415_341_646_479_339_379_50E-18,
    3.330_794_518_822_238_097_83E-17,
    -2.431_279_846_547_954_693_59E-16,
    1.715_391_285_555_133_030_61E-15,
    -1.168_533_287_799_345_168_08E-14,
    7.676_185_498_604_935_616_88E-14,
    -4.856_446_783_111_929_460_90E-13,
    2.955_052_663_129_639_834_61E-12,
    -1.726_826_291_441_555_707_23E-11,
    9.675_809_035_373_236_912_24E-11,
    -5.189_795_601_635_262_906_66E-10,
    2.659_823_724_682_386_650_35E-9,
    -1.300_025_009_986_248_042_12E-8,
    6.046_995_022_541_918_949_32E-8,
    -2.670_793_853_940_611_733_91E-7,
    1.117_387_539_120_103_718_15E-6,
    -4.416_738_358_458_750_563_59E-6,
    1.644_844_807_072_889_708_93E-5,
    -5.754_195_010_082_103_703_98E-5,
    1.885_028_850_958_416_557_29E-4,
    -5.763_755_745_385_823_658_85E-4,
    1.639_475_616_941_335_798_42E-3,
    -4.324_309_995_050_575_944_30E-3,
    1.054_646_039_459_499_831_83E-2,
    -2.373_741_480_589_946_881_56E-2,
    4.930_528_423_967_070_848_78E-2,
    -9.490_109_704_804_764_442_10E-2,
    1.716_209_015_222_087_753_49E-1,
    -3.046_826_723_431_983_986_83E-1,
    6.767_952_744_094_760_849_95E-1,
];

/// Chebyshev coefficients for `exp(-x) sqrt(x) I0(x)` on `(8, inf)` (Cephes `i0`).
#[allow(clippy::excessive_precision)] // Cephes coefficients, kept verbatim
const I0_B: [f64; 25] = [
    -7.233_180_487_874_753_954_56E-18,
    -4.830_504_485_944_182_071_26E-18,
    4.465_621_420_296_759_999_01E-17,
    3.461_222_867_697_461_093_10E-17,
    -2.827_623_980_516_583_484_94E-16,
    -3.425_485_619_677_219_134_62E-16,
    1.772_560_133_056_526_383_60E-15,
    3.811_680_669_352_622_420_75E-15,
    -9.554_846_698_828_307_648_70E-15,
    -4.150_569_347_287_222_086_63E-14,
    1.540_086_217_521_409_826_91E-14,
    3.852_778_382_742_142_701_14E-13,
    7.180_124_451_383_666_233_67E-13,
    -1.794_178_531_506_806_117_78E-12,
    -1.321_581_184_044_771_311_88E-11,
    -3.149_916_527_963_241_364_54E-11,
    1.188_914_710_784_643_834_24E-11,
    4.940_602_388_224_969_589_10E-10,
    3.396_232_025_708_386_345_15E-9,
    2.266_668_990_498_178_064_59E-8,
    2.048_918_589_469_063_741_83E-7,
    2.891_370_520_834_756_482_97E-6,
    6.889_758_346_916_823_984_26E-5,
    3.369_116_478_255_694_089_90E-3,
    8.044_904_110_141_088_316_08E-1,
];

/// Clenshaw evaluation of a Chebyshev series (Cephes `chbevl`).
fn chbevl(x: f64, coeffs: &[f64]) -> f64 {
    let mut b0 = coeffs[0];
    let mut b1 = 0.0;
    let mut b2 = 0.0;
    for &c in &coeffs[1..] {
        b2 = b1;
        b1 = b0;
        b0 = x * b1 - b2 + c;
    }
    0.5 * (b0 - b2)
}

/// Modified Bessel function of the first kind, order 0 (Cephes algorithm, matching
/// `numpy.i0` / `scipy.special.i0` to round-off).
fn bessel_i0(x: f64) -> f64 {
    let x = x.abs();
    if x <= 8.0 {
        x.exp() * chbevl(x / 2.0 - 2.0, &I0_A)
    } else {
        x.exp() * chbevl(32.0 / x - 2.0, &I0_B) / x.sqrt()
    }
}

/// Zero-pads `data` on both sides to `size` samples, centred as `librosa.util.pad_center`
/// (left pad `(size - n) / 2`, the extra sample of an odd difference goes to the right).
///
/// Callers guarantee `data.len() <= size`.
pub(crate) fn pad_center(data: &[f64], size: usize) -> Vec<f64> {
    let lpad = (size - data.len()) / 2;
    let mut out = vec![0.0; size];
    out[lpad..lpad + data.len()].copy_from_slice(data);
    out
}

/// Computes the sum-squared envelope of a window over `n_frames` overlapping frames.
///
/// Equivalent to `librosa.filters.window_sumsquare(window=window, n_frames=n_frames,
/// hop_length=hop_length, win_length=win_length, n_fft=n_fft, dtype=numpy.float64)`.
/// The window (periodic, as librosa's `get_window` default) is squared, zero-padded to
/// `n_fft` and overlap-added at multiples of `hop_length`. Dividing an overlap-add
/// resynthesis by this envelope is what makes the inverse STFT exact.
///
/// # Arguments
///
/// * `window` - The analysis window.
/// * `n_frames` - Number of frames (at least 1).
/// * `hop_length` - Samples between successive frames.
/// * `win_length` - Window length; `None` means `n_fft`. Must not exceed `n_fft`.
/// * `n_fft` - Frame length.
///
/// # Returns
///
/// The envelope, of length `n_fft + hop_length * (n_frames - 1)`.
///
/// # Errors
///
/// [`ParameterError`] if `n_frames` is 0, `win_length > n_fft`, the window is invalid,
/// or the envelope (or `n_fft`) would exceed 2^31 samples.
///
/// # Differences from librosa
///
/// librosa's `norm` argument (normalising the window before squaring) is not supported,
/// and the output is always `f64` (librosa defaults to `float32`).
///
/// # Examples
///
/// ```rust
/// use audio_samples::nzu;
/// use audio_samples::operations::filterbanks::{Window, window_sumsquare};
///
/// // A periodic Hann window at 75 % overlap sums to a constant 1.5 in the interior.
/// let wss = window_sumsquare(&Window::Hann, 10, nzu!(256), None, nzu!(1024)).unwrap();
/// assert_eq!(wss.len(), 1024 + 256 * 9);
/// assert!((wss[1500] - 1.5).abs() < 1e-12);
/// ```
pub fn window_sumsquare(
    window: &Window,
    n_frames: usize,
    hop_length: NonZeroUsize,
    win_length: Option<NonZeroUsize>,
    n_fft: NonZeroUsize,
) -> AudioSampleResult<Array1<f64>> {
    if n_frames == 0 {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "n_frames",
            "must be at least 1",
        )));
    }
    check_derived_len(
        "n_frames",
        "the window envelope",
        n_fft.get() as f64 + hop_length.get() as f64 * (n_frames - 1) as f64,
        MAX_SIGNAL_ELEMENTS,
    )?;
    let win_sq = padded_window(window, win_length, n_fft)?.mapv(|w| w * w);
    Ok(window_sumsquare_from_squared(
        win_sq.as_slice().expect("freshly allocated"),
        n_frames,
        hop_length.get(),
    ))
}

/// Overlap-adds an already squared, `n_fft`-long window (librosa `__window_ss_fill`).
pub(crate) fn window_sumsquare_from_squared(
    win_sq: &[f64],
    n_frames: usize,
    hop: usize,
) -> Array1<f64> {
    let n_fft = win_sq.len();
    let n = n_fft + hop * (n_frames - 1);
    let mut x = vec![0.0; n];
    for i in 0..n_frames {
        let start = i * hop;
        let len = n_fft.min(n - start);
        for (xs, &w) in x[start..start + len].iter_mut().zip(&win_sq[..len]) {
            *xs += w;
        }
    }
    Array1::from(x)
}

/// The periodic `win_length` window zero-padded (centred) to `n_fft`, as used by
/// librosa's STFT and ISTFT.
pub(crate) fn padded_window(
    window: &Window,
    win_length: Option<NonZeroUsize>,
    n_fft: NonZeroUsize,
) -> AudioSampleResult<Array1<f64>> {
    let win_length = win_length.map_or(n_fft.get(), NonZeroUsize::get);
    check_derived_len("n_fft", "a window", n_fft.get() as f64, MAX_SIGNAL_ELEMENTS)?;
    if win_length > n_fft.get() {
        return Err(AudioSampleError::Parameter(ParameterError::out_of_range(
            "win_length",
            win_length,
            1,
            n_fft.get(),
            "the window cannot be longer than the FFT frame (n_fft)",
        )));
    }
    let w = get_window(window, win_length, true)?;
    Ok(Array1::from(pad_center(
        w.as_slice().expect("freshly allocated"),
        n_fft.get(),
    )))
}

/// Computes the equivalent noise bandwidth of a window, in FFT bins.
///
/// Equivalent to `librosa.filters.window_bandwidth(window, n)`:
/// `n * sum(w^2) / sum(w)^2` for the periodic window of length `n`. A rectangular
/// window has bandwidth 1; a Hann window 1.5.
///
/// # Arguments
///
/// * `window` - The window. A [`Window::Custom`] must have exactly `n` coefficients.
/// * `n` - The window length to evaluate (librosa's default is 1000).
///
/// # Returns
///
/// The normalised equivalent noise bandwidth.
///
/// # Errors
///
/// [`ParameterError`] if the window is invalid for length `n`.
///
/// # Differences from librosa
///
/// For named windows librosa returns a value from a hard-coded table
/// (`librosa.filters.WINDOW_BANDWIDTHS`) and ignores `n`; those table values differ
/// from the formula at `n = 1000` by up to ~1e-4 relative (e.g. Blackman 1.72697 vs
/// 1.72676). This function always evaluates the formula.
///
/// # Examples
///
/// ```rust
/// use audio_samples::nzu;
/// use audio_samples::operations::filterbanks::{Window, window_bandwidth};
///
/// let bw = window_bandwidth(&Window::Hann, nzu!(1000)).unwrap();
/// assert!((bw - 1.5).abs() < 1e-12);
/// ```
pub fn window_bandwidth(window: &Window, n: NonZeroUsize) -> AudioSampleResult<f64> {
    let w = get_window(window, n.get(), true)?;
    let sum: f64 = w.sum();
    let sum_sq: f64 = w.iter().map(|x| x * x).sum();
    Ok(n.get() as f64 * sum_sq / (sum * sum + f64::MIN_POSITIVE))
}

// =============================================================================
// Mel filterbank
// =============================================================================

/// Normalisation applied to each mel filter (librosa's `norm` argument).
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::filterbanks::MelFilterNorm;
///
/// assert_eq!(MelFilterNorm::default(), MelFilterNorm::Slaney);
/// let l1 = MelFilterNorm::Lp(1.0); // each filter sums to 1
/// # let _ = l1;
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum MelFilterNorm {
    /// No normalisation: triangles with unit peak (librosa `norm=None`).
    None,
    /// Divide each triangle by half its bandwidth in Hz, so each band has
    /// approximately constant energy (librosa `norm='slaney'`, the default).
    #[default]
    Slaney,
    /// Divide each filter by its `p`-norm (librosa `norm=p`, via `librosa.util.normalize`).
    /// `p` may be any positive number, `f64::INFINITY` (maximum absolute value),
    /// `f64::NEG_INFINITY` (minimum absolute value) or `0.0` (number of non-zero
    /// weights). A filter whose norm is zero is left unscaled.
    Lp(f64),
}

/// Configuration of a mel filterbank, mirroring `librosa.filters.mel`.
///
/// ## Purpose
///
/// Describes how FFT bins are pooled into mel bands: how many bands, which
/// frequency range they span, which mel scale spaces them and how each
/// triangular filter is normalised.
///
/// ## Intended usage
///
/// Pass to [`mel_filterbank`], or embed in a
/// [`MelSpectrogramConfig`](crate::operations::spectrum::MelSpectrogramConfig).
/// `Default` reproduces librosa's defaults: 128 bands, 0 Hz to Nyquist, Slaney
/// scale (`htk=False`), Slaney normalisation.
///
/// ## Invariants
///
/// Checked by [`MelFilterbankConfig::validate`]: `fmin >= 0`, the resolved
/// `fmax` is finite and greater than `fmin`, and an `Lp` norm has a valid `p`.
/// `fmax` above the Nyquist frequency is allowed (as in librosa), but the bands
/// above Nyquist are then empty.
///
/// # Examples
///
/// ```rust
/// use audio_samples::nzu;
/// use audio_samples::operations::filterbanks::{MelFilterNorm, MelFilterbankConfig, MelScale};
///
/// let cfg = MelFilterbankConfig::new()
///     .with_n_mels(nzu!(40))
///     .with_fmin(20.0)
///     .with_fmax(Some(8000.0))
///     .with_scale(MelScale::Htk)
///     .with_norm(MelFilterNorm::None);
/// assert!(cfg.validate(16000.0).is_ok());
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct MelFilterbankConfig {
    /// Number of mel bands (rows of the filterbank). More bands give finer
    /// frequency resolution but narrower, noisier bands. librosa default: 128.
    pub n_mels: NonZeroUsize,
    /// Lower edge of the lowest band, in Hz. Must be `>= 0`. librosa default: 0.
    pub fmin: f64,
    /// Upper edge of the highest band, in Hz; `None` means the Nyquist frequency
    /// (`sample_rate / 2`, librosa's default). Must exceed `fmin`.
    pub fmax: Option<f64>,
    /// Mel scale used to space the band edges. librosa default: Slaney (`htk=False`).
    pub scale: MelScale,
    /// Per-filter normalisation. librosa default: Slaney.
    pub norm: MelFilterNorm,
}

impl Default for MelFilterbankConfig {
    fn default() -> Self {
        Self {
            n_mels: crate::nzu!(128),
            fmin: 0.0,
            fmax: None,
            scale: MelScale::Slaney,
            norm: MelFilterNorm::Slaney,
        }
    }
}

impl MelFilterbankConfig {
    /// Returns the librosa-default configuration (same as [`Default`]).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the number of mel bands.
    #[must_use]
    pub fn with_n_mels(mut self, n_mels: NonZeroUsize) -> Self {
        self.n_mels = n_mels;
        self
    }

    /// Sets the lower frequency edge in Hz.
    #[must_use]
    pub fn with_fmin(mut self, fmin: f64) -> Self {
        self.fmin = fmin;
        self
    }

    /// Sets the upper frequency edge in Hz (`None` = Nyquist).
    #[must_use]
    pub fn with_fmax(mut self, fmax: Option<f64>) -> Self {
        self.fmax = fmax;
        self
    }

    /// Sets the mel scale.
    #[must_use]
    pub fn with_scale(mut self, scale: MelScale) -> Self {
        self.scale = scale;
        self
    }

    /// Sets the filter normalisation.
    #[must_use]
    pub fn with_norm(mut self, norm: MelFilterNorm) -> Self {
        self.norm = norm;
        self
    }

    /// The upper frequency edge for `sample_rate` (resolving `None` to Nyquist).
    #[must_use]
    pub fn resolved_fmax(&self, sample_rate: f64) -> f64 {
        self.fmax.unwrap_or(sample_rate / 2.0)
    }

    /// Checks the configuration for a given sample rate.
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if `sample_rate` is not finite and positive, `fmin` is
    /// negative or non-finite, the resolved `fmax` is not finite or not above
    /// `fmin`, or an `Lp` norm has a negative or `NaN` exponent.
    pub fn validate(&self, sample_rate: f64) -> AudioSampleResult<()> {
        if !(sample_rate.is_finite() && sample_rate > 0.0) {
            return Err(AudioSampleError::Parameter(ParameterError::out_of_range(
                "sample_rate",
                sample_rate,
                "> 0",
                "finite",
                "sample rate must be finite and positive",
            )));
        }
        if !(self.fmin.is_finite() && self.fmin >= 0.0) {
            return Err(AudioSampleError::Parameter(ParameterError::out_of_range(
                "fmin",
                self.fmin,
                0,
                "finite",
                "lower mel edge must be finite and non-negative",
            )));
        }
        let fmax = self.resolved_fmax(sample_rate);
        if !(fmax.is_finite() && fmax > self.fmin) {
            return Err(AudioSampleError::Parameter(ParameterError::out_of_range(
                "fmax",
                fmax,
                format!("> fmin ({})", self.fmin),
                "finite",
                "upper mel edge must be finite and above fmin",
            )));
        }
        if let MelFilterNorm::Lp(p) = self.norm {
            if p.is_nan() || (p.is_finite() && p < 0.0) {
                return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                    "norm",
                    "Lp norm exponent must be > 0, 0, +inf or -inf",
                )));
            }
        }
        Ok(())
    }
}

/// Builds a mel filterbank matrix, equivalent to `librosa.filters.mel`.
///
/// Each row is a triangular filter over the FFT bins: band `i` rises from mel edge
/// `i` to edge `i + 1` and falls to edge `i + 2`, where the `n_mels + 2` edges are
/// spaced uniformly on the chosen mel scale between `fmin` and `fmax`. Multiplying
/// the filterbank with a power (or magnitude) spectrogram of matching `n_fft` gives a
/// mel spectrogram.
///
/// # Arguments
///
/// * `sample_rate` - Sampling rate of the analysed signal, in Hz.
/// * `n_fft` - FFT length of the spectrogram the filterbank will be applied to.
/// * `config` - Number of bands, frequency range, scale and normalisation.
///
/// # Returns
///
/// An `(n_mels, 1 + n_fft / 2)` matrix of non-negative weights (`f64`).
///
/// # Errors
///
/// [`ParameterError`] from [`MelFilterbankConfig::validate`], or if the
/// `n_mels x (1 + n_fft / 2)` matrix would exceed 2^28 elements.
///
/// # Differences from librosa
///
/// The weights are computed and returned in `f64`; librosa's default
/// `dtype=numpy.float32` rounds them to single precision (relative differences
/// around 1e-7). librosa warns about empty filters (too many bands for the
/// frequency resolution); this function returns them silently as all-zero rows.
///
/// # Examples
///
/// ```rust
/// use audio_samples::nzu;
/// use audio_samples::operations::filterbanks::{MelFilterbankConfig, mel_filterbank};
///
/// let fb = mel_filterbank(16000.0, nzu!(512), &MelFilterbankConfig::new().with_n_mels(nzu!(40)))
///     .unwrap();
/// assert_eq!(fb.dim(), (40, 257));
/// assert!(fb.iter().all(|&w| w >= 0.0));
/// ```
///
/// # References
///
/// M. Slaney, *Auditory Toolbox*, Interval Research Corporation, 1998;
/// librosa `filters.mel`.
pub fn mel_filterbank(
    sample_rate: f64,
    n_fft: NonZeroUsize,
    config: &MelFilterbankConfig,
) -> AudioSampleResult<Array2<f64>> {
    config.validate(sample_rate)?;
    let n_mels = config.n_mels.get();
    let n_fft = n_fft.get();
    let n_bins = 1 + n_fft / 2;
    check_derived_len(
        "n_mels",
        "a mel filterbank",
        n_mels as f64 * n_bins as f64,
        MAX_TABLE_ELEMENTS,
    )?;
    let fmax = config.resolved_fmax(sample_rate);

    // numpy.fft.rfftfreq(n_fft, d=1/sr): k * (1 / (n_fft * d)).
    let d = 1.0 / sample_rate;
    let val = 1.0 / (n_fft as f64 * d);
    let fftfreqs: Vec<f64> = (0..n_bins).map(|k| k as f64 * val).collect();

    let mel_f = mel_frequencies_with_scale(n_mels + 2, config.fmin, fmax, config.scale);
    let fdiff: Vec<f64> = mel_f.windows(2).map(|w| w[1] - w[0]).collect();

    let mut weights = Array2::<f64>::zeros((n_mels, n_bins));
    for (i, mut row) in weights.outer_iter_mut().enumerate() {
        for (k, w) in row.iter_mut().enumerate() {
            let lower = -(mel_f[i] - fftfreqs[k]) / fdiff[i];
            let upper = (mel_f[i + 2] - fftfreqs[k]) / fdiff[i + 1];
            *w = 0.0f64.max(lower.min(upper));
        }
    }

    match config.norm {
        MelFilterNorm::None => {}
        MelFilterNorm::Slaney => {
            for (i, mut row) in weights.outer_iter_mut().enumerate() {
                let enorm = 2.0 / (mel_f[i + 2] - mel_f[i]);
                row.mapv_inplace(|w| w * enorm);
            }
        }
        MelFilterNorm::Lp(p) => {
            for mut row in weights.outer_iter_mut() {
                let length = lp_norm(row.iter().copied(), p);
                // librosa.util.normalize(fill=None): rows below `tiny` are left as is.
                let length = if length < f64::MIN_POSITIVE {
                    1.0
                } else {
                    length
                };
                row.mapv_inplace(|w| w / length);
            }
        }
    }
    Ok(weights)
}

/// The `librosa.util.normalize` "length" of a vector for norm `p`.
fn lp_norm(values: impl Iterator<Item = f64>, p: f64) -> f64 {
    let mags = values.map(f64::abs);
    if p == f64::INFINITY {
        mags.fold(f64::NEG_INFINITY, f64::max)
    } else if p == f64::NEG_INFINITY {
        mags.fold(f64::INFINITY, f64::min)
    } else if p == 0.0 {
        mags.filter(|&m| m > 0.0).count() as f64
    } else {
        mags.map(|m| m.powf(p)).sum::<f64>().powf(1.0 / p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extreme_window_sizes_are_parameter_errors() {
        let is_param =
            |r: AudioSampleResult<Array1<f64>>| matches!(r, Err(AudioSampleError::Parameter(_)));
        let big = 1usize << 40;
        assert!(is_param(get_window(&Window::Hann, big, true)));
        assert!(window_bandwidth(&Window::Hann, NonZeroUsize::new(big).unwrap()).is_err());
        assert!(is_param(window_sumsquare(
            &Window::Hann,
            big,
            nzu!(512),
            None,
            nzu!(2048)
        )));
        assert!(is_param(window_sumsquare(
            &Window::Hann,
            3,
            NonZeroUsize::new(usize::MAX).unwrap(),
            None,
            nzu!(2048)
        )));
        assert!(is_param(padded_window(
            &Window::Hann,
            None,
            NonZeroUsize::new(big).unwrap()
        )));
        let huge_mels = MelFilterbankConfig::new().with_n_mels(NonZeroUsize::new(1 << 30).unwrap());
        assert!(matches!(
            mel_filterbank(22050.0, nzu!(2048), &huge_mels),
            Err(AudioSampleError::Parameter(_))
        ));
        assert!(get_window(&Window::Hann, 1024, true).is_ok());
    }

    #[test]
    fn hann_periodic_and_symmetric() {
        let p = get_window(&Window::Hann, 4, true).unwrap();
        let expected = [0.0, 0.5, 1.0, 0.5];
        for (a, b) in p.iter().zip(expected) {
            assert!((a - b).abs() < 1e-15);
        }
        let s = get_window(&Window::Hann, 5, false).unwrap();
        let expected = [0.0, 0.5, 1.0, 0.5, 0.0];
        for (a, b) in s.iter().zip(expected) {
            assert!((a - b).abs() < 1e-15);
        }
    }

    #[test]
    fn short_and_degenerate_windows() {
        assert_eq!(get_window(&Window::Hann, 0, true).unwrap().len(), 0);
        assert_eq!(
            get_window(&Window::Kaiser { beta: 3.0 }, 1, true)
                .unwrap()
                .to_vec(),
            vec![1.0]
        );
        assert!(get_window(&Window::Gaussian { std: 0.0 }, 8, true).is_err());
        assert!(get_window(&Window::Kaiser { beta: f64::NAN }, 8, true).is_err());
        assert!(get_window(&Window::Custom(vec![1.0, 2.0]), 3, true).is_err());
        assert!(get_window(&Window::Custom(vec![1.0, f64::NAN]), 2, true).is_err());
    }

    #[test]
    fn triangular_windows_match_scipy_definitions() {
        // scipy.signal.windows.triang(4) = [0.25, 0.75, 0.75, 0.25]; triang(5) = [1/3, 2/3, 1, 2/3, 1/3]
        let t4 = get_window(&Window::Triangular, 4, false).unwrap();
        assert_eq!(t4.to_vec(), vec![0.25, 0.75, 0.75, 0.25]);
        let t5 = get_window(&Window::Triangular, 5, false).unwrap();
        for (a, b) in t5
            .iter()
            .zip([1.0 / 3.0, 2.0 / 3.0, 1.0, 2.0 / 3.0, 1.0 / 3.0])
        {
            assert!((a - b).abs() < 1e-15);
        }
        // bartlett(5) = [0, 0.5, 1, 0.5, 0]
        let b5 = get_window(&Window::Bartlett, 5, false).unwrap();
        assert_eq!(b5.to_vec(), vec![0.0, 0.5, 1.0, 0.5, 0.0]);
    }

    #[test]
    fn bessel_i0_known_values() {
        assert!((bessel_i0(0.0) - 1.0).abs() < 1e-15);
        // I0(1) = 1.2660658777520082, I0(10) = 2815.716628466254
        assert!((bessel_i0(1.0) - 1.266_065_877_752_008_2).abs() < 1e-14);
        assert!((bessel_i0(10.0) / 2_815.716_628_466_254 - 1.0).abs() < 1e-14);
    }

    #[test]
    fn window_sumsquare_constant_overlap() {
        let wss = window_sumsquare(&Window::Hann, 20, nzu!(512), None, nzu!(2048)).unwrap();
        assert_eq!(wss.len(), 2048 + 512 * 19);
        for &v in wss.slice(ndarray::s![2048..2048 + 512 * 15]) {
            assert!((v - 1.5).abs() < 1e-12);
        }
        assert!(window_sumsquare(&Window::Hann, 0, nzu!(512), None, nzu!(2048)).is_err());
        assert!(
            window_sumsquare(&Window::Hann, 2, nzu!(512), Some(nzu!(4096)), nzu!(2048)).is_err()
        );
    }

    #[test]
    fn padded_window_is_centred() {
        let w = padded_window(&Window::Rectangular, Some(nzu!(3)), nzu!(8)).unwrap();
        assert_eq!(w.to_vec(), vec![0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn window_bandwidths() {
        assert!((window_bandwidth(&Window::Rectangular, nzu!(64)).unwrap() - 1.0).abs() < 1e-12);
        assert!((window_bandwidth(&Window::Hann, nzu!(1000)).unwrap() - 1.5).abs() < 1e-12);
    }

    #[test]
    fn mel_filterbank_shapes_and_norms() {
        let cfg = MelFilterbankConfig::new()
            .with_n_mels(nzu!(10))
            .with_norm(MelFilterNorm::None);
        let fb = mel_filterbank(8000.0, nzu!(256), &cfg).unwrap();
        assert_eq!(fb.dim(), (10, 129));
        // Unnormalised triangles peak at most 1.
        assert!(fb.iter().all(|&w| (0.0..=1.0).contains(&w)));

        let l1 = mel_filterbank(
            8000.0,
            nzu!(256),
            &cfg.clone().with_norm(MelFilterNorm::Lp(1.0)),
        )
        .unwrap();
        for row in l1.outer_iter() {
            assert!((row.sum() - 1.0).abs() < 1e-12);
        }
        let linf = mel_filterbank(
            8000.0,
            nzu!(256),
            &cfg.with_norm(MelFilterNorm::Lp(f64::INFINITY)),
        )
        .unwrap();
        for row in linf.outer_iter() {
            assert!((row.iter().copied().fold(0.0, f64::max) - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn mel_filterbank_rejects_bad_ranges() {
        let fb = |cfg: MelFilterbankConfig| mel_filterbank(16000.0, nzu!(512), &cfg);
        assert!(fb(MelFilterbankConfig::new().with_fmin(-1.0)).is_err());
        assert!(fb(MelFilterbankConfig::new().with_fmin(9000.0)).is_err());
        assert!(fb(MelFilterbankConfig::new().with_fmax(Some(f64::INFINITY))).is_err());
        assert!(fb(MelFilterbankConfig::new().with_norm(MelFilterNorm::Lp(-2.0))).is_err());
        assert!(mel_filterbank(0.0, nzu!(512), &MelFilterbankConfig::new()).is_err());
        // fmax above Nyquist is allowed, as in librosa.
        assert!(fb(MelFilterbankConfig::new().with_fmax(Some(12000.0))).is_ok());
    }
}
