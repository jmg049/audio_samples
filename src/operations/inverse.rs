//! Inversion of mel spectrograms and MFCCs to spectrograms and audio.
//!
//! # What
//!
//! - [`mel_to_stft`] estimates an STFT magnitude from a mel spectrogram by non-negative
//!   least squares.
//! - [`mel_to_audio`] continues to audio with fast Griffin-Lim.
//! - [`mfcc_to_mel`] inverts the DCT, the liftering and the decibel scale of MFCCs.
//! - [`mfcc_to_audio`] chains the two.
//!
//! # Why
//!
//! A mel spectrogram or MFCC discards phase and most of the frequency resolution, so
//! these are estimates. They are useful for listening to what a feature retains, and
//! for vocoder baselines.
//!
//! # How
//!
//! Inputs have shape `(channels, bands, frames)`. Output precision follows the input.
//! The configs hold every parameter; [`MelInversionConfig::new`] needs the sample rate
//! because the filterbank depends on it.
//!
//! ```
//! use audio_samples::nzu;
//! use audio_samples::operations::inverse::{MelInversionConfig, mel_to_stft};
//! use ndarray::Array3;
//!
//! let mel = Array3::<f32>::from_elem((1, 40, 3), 1.0);
//! let cfg = MelInversionConfig::new(22050.0).with_n_fft(nzu!(512));
//! let s = mel_to_stft(mel.view(), &cfg).unwrap();
//! assert_eq!(s.dim(), (1, 257, 3));
//! ```
//!
//! References: D. Griffin and J. Lim, *Signal estimation from modified short-time
//! Fourier transform*, IEEE TASSP 32, 1984; N. Perraudin, P. Balazs and P. L.
//! Søndergaard, *A fast Griffin-Lim algorithm*, WASPAA 2013.

use std::num::NonZeroUsize;

use ndarray::linalg::general_mat_mul;
use ndarray::{Array2, Array3, ArrayView3, Axis, s};

use crate::audio_float::float_rows_to_audio;
use crate::operations::effects::{GriffinLimConfig, griffinlim};
use crate::operations::filterbanks::{MelFilterbankConfig, mel_filterbank};
use crate::operations::linalg::{NnlsConfig, NnlsSolver};
use crate::operations::spectrum::{DctNorm, DctType};
use crate::repr::SampleRate;
use crate::{AudioFloat, AudioSampleResult, AudioSamples, ParameterError, StandardSample};

fn invalid<T>(parameter: &str, reason: impl Into<String>) -> AudioSampleResult<T> {
    Err(ParameterError::invalid_value(parameter, reason.into()).into())
}

fn of<F: AudioFloat>(v: f64) -> F {
    <F as num_traits::NumCast>::from(v).unwrap_or_else(F::nan)
}

/// Parameters of [`mel_to_stft`] and [`mel_to_audio`].
#[derive(Debug, Clone, PartialEq)]
pub struct MelInversionConfig {
    /// Sample rate the mel spectrogram was computed at, in Hz.
    pub sample_rate: f64,
    /// FFT size of the spectrogram to estimate. Default 2048.
    pub n_fft: NonZeroUsize,
    /// Exponent that was applied to the magnitude: 2 for power, 1 for magnitude. Default 2.
    pub power: f64,
    /// The filterbank. `n_mels` is taken from the input's shape.
    pub mel: MelFilterbankConfig,
    /// Stopping rule of the least-squares solver.
    pub nnls: NnlsConfig,
}

impl MelInversionConfig {
    /// A configuration for the given sample rate, with an FFT size of 2048, power 2 and
    /// the default filterbank.
    #[must_use]
    pub fn new(sample_rate: f64) -> Self {
        Self {
            sample_rate,
            n_fft: crate::nzu!(2048),
            power: 2.0,
            mel: MelFilterbankConfig::new(),
            nnls: NnlsConfig::new(),
        }
    }

    /// Settings matching librosa 1.0's defaults, including a sample rate of 22050 Hz.
    #[must_use]
    pub fn librosa() -> Self {
        Self::new(22050.0)
    }

    /// Sets the FFT size.
    #[must_use]
    pub fn with_n_fft(mut self, n_fft: NonZeroUsize) -> Self {
        self.n_fft = n_fft;
        self
    }

    /// Sets the magnitude exponent.
    #[must_use]
    pub fn with_power(mut self, power: f64) -> Self {
        self.power = power;
        self
    }

    /// Sets the filterbank.
    #[must_use]
    pub fn with_mel(mut self, mel: MelFilterbankConfig) -> Self {
        self.mel = mel;
        self
    }

    /// Sets the stopping rule of the least-squares solver.
    #[must_use]
    pub fn with_nnls(mut self, nnls: NnlsConfig) -> Self {
        self.nnls = nnls;
        self
    }

    /// Checks the configuration.
    ///
    /// # Errors
    /// [`ParameterError`] if `power` or the sample rate is not positive and finite.
    pub fn validate(&self) -> AudioSampleResult<()> {
        if !(self.power.is_finite() && self.power > 0.0) {
            return invalid("power", "must be positive and finite");
        }
        if !(self.sample_rate.is_finite() && self.sample_rate > 0.0) {
            return invalid("sample_rate", "must be positive and finite");
        }
        self.nnls.validate()
    }
}

/// Estimates an STFT magnitude from a mel spectrogram.
///
/// Solves `mel = basis * x` with `x >= 0` for each frame, and returns `x^(1 / power)`.
///
/// # Arguments
/// - `mel`: mel spectrogram, `(channels, n_mels, frames)`.
/// - `config`: the sample rate, FFT size, exponent and filterbank.
///
/// # Returns
/// The magnitude, `(channels, 1 + n_fft / 2, frames)`.
///
/// # Errors
/// [`ParameterError`] for an invalid configuration or non-finite input.
///
/// # Examples
/// See the [module documentation](self).
pub fn mel_to_stft<F: AudioFloat>(
    mel: ArrayView3<'_, F>,
    config: &MelInversionConfig,
) -> AudioSampleResult<Array3<F>> {
    config.validate()?;
    let (channels, n_mels, frames) = mel.dim();
    let n_mels = NonZeroUsize::new(n_mels)
        .ok_or_else(|| ParameterError::invalid_value("mel", "has no mel bands".to_string()))?;
    let basis = mel_filterbank(
        config.sample_rate,
        config.n_fft,
        &config.mel.clone().with_n_mels(n_mels),
    )?
    .mapv(of::<F>);
    let solver = NnlsSolver::new(basis.view(), &config.nnls)?;
    let bins = basis.ncols();
    let mut out = Array3::<F>::zeros((channels, bins, frames));
    let root = 1.0 / config.power;
    for (mut o, m) in out.axis_iter_mut(Axis(0)).zip(mel.axis_iter(Axis(0))) {
        let mut x = solver.solve(m)?;
        if config.power == 2.0 {
            x.mapv_inplace(num_traits::Float::sqrt);
        } else if config.power != 1.0 {
            let e = of::<F>(root);
            x.mapv_inplace(|v| v.powf(e));
        }
        o.assign(&x);
    }
    Ok(out)
}

/// Estimates audio from a mel spectrogram: [`mel_to_stft`], then fast Griffin-Lim.
///
/// # Arguments
/// - `mel`: mel spectrogram, `(channels, n_mels, frames)`.
/// - `config`: as for [`mel_to_stft`].
/// - `griffin_lim`: iterations, phase initialisation and STFT parameters. Its FFT size
///   must equal `config.n_fft`.
///
/// # Returns
/// `(channels, samples)` audio.
///
/// # Errors
/// [`ParameterError`] for an invalid configuration, mismatched FFT sizes or non-finite
/// input.
///
/// # Examples
/// ```
/// use audio_samples::nzu;
/// use audio_samples::operations::effects::{GriffinLimConfig, GriffinLimInit};
/// use audio_samples::operations::inverse::{MelInversionConfig, mel_to_audio};
/// use audio_samples::operations::spectrum::StftConfig;
/// use ndarray::Array3;
///
/// let mel = Array3::<f64>::from_elem((1, 40, 4), 1.0);
/// let cfg = MelInversionConfig::new(22050.0).with_n_fft(nzu!(512));
/// let gl = GriffinLimConfig::new()
///     .with_n_iter(2)
///     .with_init(GriffinLimInit::ZeroPhase)
///     .with_stft(StftConfig::new().with_n_fft(nzu!(512)));
/// let y = mel_to_audio(mel.view(), &cfg, &gl).unwrap();
/// assert_eq!(y.nrows(), 1);
/// ```
pub fn mel_to_audio<F: AudioFloat>(
    mel: ArrayView3<'_, F>,
    config: &MelInversionConfig,
    griffin_lim: &GriffinLimConfig,
) -> AudioSampleResult<Array2<F>> {
    if griffin_lim.stft.n_fft != config.n_fft {
        return invalid(
            "griffin_lim",
            format!(
                "its FFT size {} differs from the inversion's {}",
                griffin_lim.stft.n_fft, config.n_fft
            ),
        );
    }
    let s = mel_to_stft(mel, config)?;
    griffinlim(s.view(), griffin_lim)
}

/// [`mel_to_audio`] returning audio of the sample type `T`.
///
/// # Arguments
/// - `mel`: mel spectrogram in `T`'s float type.
/// - `config`, `griffin_lim`: as for [`mel_to_audio`].
/// - `sample_rate`: the rate of the result.
///
/// # Returns
/// Mono audio for one channel, otherwise multi-channel audio; integer types are rounded
/// and saturated.
///
/// # Errors
/// As for [`mel_to_audio`].
pub fn mel_to_audio_samples<T: StandardSample>(
    mel: ArrayView3<'_, T::Float>,
    config: &MelInversionConfig,
    griffin_lim: &GriffinLimConfig,
    sample_rate: SampleRate,
) -> AudioSampleResult<AudioSamples<'static, T>> {
    let y = mel_to_audio(mel, config, griffin_lim)?;
    float_rows_to_audio(y, sample_rate, "mel_to_audio")
}

/// Parameters of [`mfcc_to_mel`] and [`mfcc_to_audio`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MfccInversionConfig {
    /// Mel bands to recover. Default 128.
    pub n_mels: NonZeroUsize,
    /// The DCT the coefficients were computed with. Default DCT-II.
    pub dct_type: DctType,
    /// The DCT's normalisation. Default orthonormal.
    pub dct_norm: DctNorm,
    /// The reference of the decibel scale. Default 1.
    pub reference: f64,
    /// The liftering coefficient to undo; 0 for none. Default 0.
    pub lifter: f64,
}

impl MfccInversionConfig {
    /// 128 bands, orthonormal DCT-II, reference 1, no liftering.
    #[must_use]
    pub fn new() -> Self {
        Self {
            n_mels: crate::nzu!(128),
            dct_type: DctType::Two,
            dct_norm: DctNorm::Ortho,
            reference: 1.0,
            lifter: 0.0,
        }
    }

    /// Settings matching librosa 1.0's defaults; currently equal to [`new`](Self::new).
    #[must_use]
    pub fn librosa() -> Self {
        Self::new()
    }

    /// Sets the number of mel bands.
    #[must_use]
    pub fn with_n_mels(mut self, n_mels: NonZeroUsize) -> Self {
        self.n_mels = n_mels;
        self
    }

    /// Sets the DCT type.
    #[must_use]
    pub fn with_dct_type(mut self, dct_type: DctType) -> Self {
        self.dct_type = dct_type;
        self
    }

    /// Sets the DCT normalisation.
    #[must_use]
    pub fn with_dct_norm(mut self, dct_norm: DctNorm) -> Self {
        self.dct_norm = dct_norm;
        self
    }

    /// Sets the decibel reference.
    #[must_use]
    pub fn with_reference(mut self, reference: f64) -> Self {
        self.reference = reference;
        self
    }

    /// Sets the liftering coefficient.
    #[must_use]
    pub fn with_lifter(mut self, lifter: f64) -> Self {
        self.lifter = lifter;
        self
    }

    /// Checks the configuration.
    ///
    /// # Errors
    /// [`ParameterError`] if `lifter` is negative or not finite, the reference is not
    /// finite, or DCT-I is asked for fewer than 2 bands.
    pub fn validate(&self) -> AudioSampleResult<()> {
        if !(self.lifter.is_finite() && self.lifter >= 0.0) {
            return invalid("lifter", "must be a non-negative number");
        }
        if !self.reference.is_finite() {
            return invalid("reference", "must be finite");
        }
        if matches!(self.dct_type, DctType::One) && self.n_mels.get() < 2 {
            return invalid("n_mels", "DCT-I needs at least 2 points");
        }
        Ok(())
    }
}

impl Default for MfccInversionConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// The first `rows` rows of the `(n, k)` matrix of the inverse DCT of length `n` restricted to
/// the first `k` coefficients.
fn inverse_dct_rows<F: AudioFloat>(
    rows: usize,
    n: usize,
    k: usize,
    dct_type: DctType,
    norm: DctNorm,
) -> Array2<F> {
    use std::f64::consts::PI;
    let nf = n as f64;
    let ortho = matches!(norm, DctNorm::Ortho);
    Array2::from_shape_fn((rows, k), |(i, c)| {
        let (i_f, c_f) = (i as f64, c as f64);
        of::<F>(match dct_type {
            DctType::Two => {
                let basis = (PI * c_f * (2.0 * i_f + 1.0) / (2.0 * nf)).cos();
                if ortho {
                    let scale = if c == 0 {
                        1.0 / nf.sqrt()
                    } else {
                        (2.0 / nf).sqrt()
                    };
                    scale * basis
                } else if c == 0 {
                    1.0 / (2.0 * nf)
                } else {
                    basis / nf
                }
            }
            DctType::Three => {
                let basis = (PI * i_f * (2.0 * c_f + 1.0) / (2.0 * nf)).cos();
                if ortho {
                    let scale = if i == 0 {
                        1.0 / nf.sqrt()
                    } else {
                        (2.0 / nf).sqrt()
                    };
                    scale * basis
                } else {
                    basis / nf
                }
            }
            DctType::One => {
                let m = nf - 1.0;
                let edge = c == 0 || c == n - 1;
                let basis = (PI * c_f * i_f / m).cos();
                if ortho {
                    let f = 1.0 / (2.0 * m).sqrt();
                    let in_scale = if edge { 2.0_f64.sqrt() } else { 2.0 };
                    let out_scale = if i == 0 || i == n - 1 {
                        1.0 / 2.0_f64.sqrt()
                    } else {
                        1.0
                    };
                    f * in_scale * basis * out_scale
                } else {
                    let w = if edge { 1.0 } else { 2.0 };
                    w * basis / (2.0 * m)
                }
            }
        })
    })
}

/// Recovers a mel power spectrogram from MFCCs.
///
/// Undoes the liftering, applies the inverse DCT with the missing coefficients taken as
/// zero, and converts from decibels to power.
///
/// # Arguments
/// - `mfcc`: coefficients, `(channels, n_mfcc, frames)`.
/// - `config`: the transform's parameters.
///
/// # Returns
/// The mel spectrogram, `(channels, n_mels, frames)`.
///
/// # Errors
/// [`ParameterError`] for an invalid configuration or non-finite input.
///
/// # Examples
/// ```
/// use audio_samples::nzu;
/// use audio_samples::operations::inverse::{MfccInversionConfig, mfcc_to_mel};
/// use ndarray::Array3;
///
/// let mfcc = Array3::<f64>::zeros((1, 13, 2));
/// let cfg = MfccInversionConfig::new().with_n_mels(nzu!(40));
/// let mel = mfcc_to_mel(mfcc.view(), &cfg).unwrap();
/// assert_eq!(mel.dim(), (1, 40, 2));
/// assert_eq!(mel[[0, 0, 0]], 1.0);
/// ```
pub fn mfcc_to_mel<F: AudioFloat>(
    mfcc: ArrayView3<'_, F>,
    config: &MfccInversionConfig,
) -> AudioSampleResult<Array3<F>> {
    config.validate()?;
    if mfcc.iter().any(|v| !v.is_finite()) {
        return invalid("mfcc", "must be finite");
    }
    let (channels, n_mfcc, frames) = mfcc.dim();
    let n_mels = config.n_mels.get();
    let used = n_mfcc.min(n_mels);
    let gains: Vec<F> = (0..used)
        .map(|k| {
            let sine = 1.0
                + config.lifter
                    * 0.5
                    * (std::f64::consts::PI * (k + 1) as f64 / config.lifter).sin();
            F::one() / (of::<F>(sine) + F::min_positive_value())
        })
        .collect::<Vec<F>>();
    let liftered = config.lifter > 0.0;
    let mut log_mel = Array3::<F>::zeros((channels, n_mels, frames));
    // The basis of a type-2 inverse is even in its even coefficients and odd in its odd ones
    // about the middle row, so half of it is enough: each half of the output holds one sum.
    let symmetric = matches!(config.dct_type, DctType::Two);
    let stored = if symmetric {
        n_mels.div_ceil(2)
    } else {
        n_mels
    };
    let basis: Array2<F> = inverse_dct_rows(stored, n_mels, used, config.dct_type, config.dct_norm);
    for (mut out, input) in log_mel.axis_iter_mut(Axis(0)).zip(mfcc.axis_iter(Axis(0))) {
        let lifted;
        let coefficients = if liftered {
            let mut copy = input.slice(s![..used, ..]).to_owned();
            for (mut row, &gain) in copy.rows_mut().into_iter().zip(&gains) {
                row.mapv_inplace(|v| v * gain);
            }
            lifted = copy;
            lifted.view()
        } else {
            input.slice(s![..used, ..])
        };
        if !symmetric {
            general_mat_mul(F::one(), &basis, &coefficients, F::zero(), &mut out);
            continue;
        }
        let (mut even, mut odd) = out.view_mut().split_at(Axis(0), stored);
        general_mat_mul(
            F::one(),
            &basis.slice(s![.., ..;2]),
            &coefficients.slice(s![..;2, ..]),
            F::zero(),
            &mut even,
        );
        if used > 1 && odd.nrows() > 0 {
            general_mat_mul(
                F::one(),
                &basis.slice(s![..odd.nrows(), 1..;2]),
                &coefficients.slice(s![1..;2, ..]),
                F::zero(),
                &mut odd,
            );
        }
        // Row r holds even + odd, and its mirror image even - odd.
        for (mut upper, mut lower) in even.rows_mut().into_iter().zip(odd.rows_mut()) {
            for (a, b) in upper.iter_mut().zip(lower.iter_mut()) {
                let (sum, difference) = (*a + *b, *a - *b);
                *a = sum;
                *b = difference;
            }
        }
        let count = odd.nrows();
        for r in 0..count / 2 {
            for c in 0..frames {
                odd.swap([r, c], [count - 1 - r, c]);
            }
        }
    }
    let tenth = of::<F>(0.1);
    log_mel.mapv_inplace(|x| x * tenth);
    if let Some(v) = log_mel
        .as_slice_memory_order_mut()
        .and_then(crate::utils::vmath::as_f32)
    {
        crate::utils::vmath::exp10_f32(v);
    } else {
        let ten = of::<F>(10.0);
        log_mel.mapv_inplace(|y| ten.powf(y));
    }
    if config.reference != 1.0 {
        let reference = of::<F>(config.reference);
        log_mel.mapv_inplace(|p| reference * p);
    }
    Ok(log_mel)
}

/// Estimates audio from MFCCs: [`mfcc_to_mel`], then [`mel_to_audio`].
///
/// # Arguments
/// - `mfcc`: coefficients, `(channels, n_mfcc, frames)`.
/// - `mfcc_config`: the transform's parameters.
/// - `mel_config`, `griffin_lim`: as for [`mel_to_audio`].
///
/// # Returns
/// `(channels, samples)` audio.
///
/// # Errors
/// As for [`mfcc_to_mel`] and [`mel_to_audio`].
pub fn mfcc_to_audio<F: AudioFloat>(
    mfcc: ArrayView3<'_, F>,
    mfcc_config: &MfccInversionConfig,
    mel_config: &MelInversionConfig,
    griffin_lim: &GriffinLimConfig,
) -> AudioSampleResult<Array2<F>> {
    let mel = mfcc_to_mel(mfcc, mfcc_config)?;
    mel_to_audio(mel.view(), mel_config, griffin_lim)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_bad_parameters() {
        let m = Array3::<f64>::zeros((1, 4, 2));
        let cfg = MelInversionConfig::new(22050.0).with_power(0.0);
        assert!(mel_to_stft(m.view(), &cfg).is_err());
        let cfg = MfccInversionConfig::new().with_lifter(-1.0);
        assert!(mfcc_to_mel(m.view(), &cfg).is_err());
        let mut nan = m;
        nan[[0, 0, 0]] = f64::NAN;
        assert!(mfcc_to_mel(nan.view(), &MfccInversionConfig::new()).is_err());
    }

    #[test]
    fn fft_sizes_must_agree() {
        let m = Array3::<f64>::from_elem((1, 8, 2), 1.0);
        let cfg = MelInversionConfig::new(22050.0).with_n_fft(crate::nzu!(256));
        assert!(mel_to_audio(m.view(), &cfg, &GriffinLimConfig::new()).is_err());
    }
}
