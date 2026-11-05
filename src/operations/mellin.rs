//! Fast Mellin (scale) transform.
//!
//! # What
//!
//! [`mellin_transform`] computes the discrete scale transform of a signal: its Fourier
//! transform after resampling on an exponential time grid.
//!
//! # Why
//!
//! The magnitude of the scale transform does not change when the signal is stretched in
//! time. Tempo-invariant rhythm descriptors use it, typically on an onset autocorrelation.
//!
//! # How
//!
//! The signal is taken to cover `[0, 1)`. It is interpolated at points spaced geometrically
//! from `t_min / n`, weighted by `x^beta`, and transformed with a real FFT. `beta = 0.5`
//! is the scale transform, whose energy is preserved; other values of `beta` give other
//! members of the Mellin family.
//!
//! ```
//! use audio_samples::operations::mellin::{MellinConfig, mellin_transform};
//! use ndarray::{Array1, Axis};
//!
//! let y = Array1::from_iter((0..256).map(|i| (-(i as f64) / 64.0).exp())).into_dyn();
//! let m = mellin_transform(y.view(), &MellinConfig::new(), Axis(0)).unwrap();
//! assert!(m.len() > 100);
//! ```
//!
//! References: L. Cohen, *The scale representation*, IEEE TSP 41, 1993; A. De Sena and
//! D. Rocchesso, *A fast Mellin and scale transform*, EURASIP JASP 2007.

use ndarray::{Array1, ArrayD, ArrayViewD, Axis, IxDyn, Zip};
use num_complex::Complex;

use crate::utils::array_ops::{InterpKind, interp1d};
use crate::{AudioFloat, AudioSampleResult, ParameterError};

fn invalid<T>(parameter: &str, reason: impl Into<String>) -> AudioSampleResult<T> {
    Err(ParameterError::invalid_value(parameter, reason.into()).into())
}

fn of<F: AudioFloat>(v: f64) -> F {
    <F as num_traits::NumCast>::from(v).unwrap_or_else(F::nan)
}

/// Parameters of [`mellin_transform`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MellinConfig {
    /// Where the exponential grid starts, in samples. Default 0.5.
    pub t_min: f64,
    /// Points on the exponential grid; `None` chooses enough to match the highest
    /// resolution of the signal. At least 3. Default `None`.
    pub n_fmt: Option<usize>,
    /// The interpolant. Default cubic.
    pub kind: InterpKind,
    /// Exponent of the weighting. 0.5 is the scale transform. Default 0.5.
    pub beta: f64,
    /// Oversampling of the grid, at least 1 when `n_fmt` is `None`. Default 1.
    pub over_sample: f64,
}

impl MellinConfig {
    /// The scale transform from half a sample, with a cubic interpolant.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            t_min: 0.5,
            n_fmt: None,
            kind: InterpKind::Cubic,
            beta: 0.5,
            over_sample: 1.0,
        }
    }

    /// librosa's defaults; currently equal to [`new`](Self::new).
    #[must_use]
    pub const fn librosa() -> Self {
        Self::new()
    }

    /// Sets where the grid starts.
    #[must_use]
    pub const fn with_t_min(mut self, t_min: f64) -> Self {
        self.t_min = t_min;
        self
    }

    /// Sets the number of grid points.
    #[must_use]
    pub const fn with_n_fmt(mut self, n_fmt: Option<usize>) -> Self {
        self.n_fmt = n_fmt;
        self
    }

    /// Sets the interpolant.
    #[must_use]
    pub const fn with_kind(mut self, kind: InterpKind) -> Self {
        self.kind = kind;
        self
    }

    /// Sets the weighting exponent.
    #[must_use]
    pub const fn with_beta(mut self, beta: f64) -> Self {
        self.beta = beta;
        self
    }

    /// Sets the oversampling.
    #[must_use]
    pub const fn with_over_sample(mut self, over_sample: f64) -> Self {
        self.over_sample = over_sample;
        self
    }
}

impl Default for MellinConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// The Mellin transform along one axis.
///
/// # Arguments
/// - `y`: finite samples, at least 3 along `axis`.
/// - `config`: the grid, the interpolant and the weighting.
/// - `axis`: the time axis.
///
/// # Returns
/// Complex coefficients, `n_fmt / 2 + 1` along `axis`.
///
/// # Errors
/// [`ParameterError`] if `axis` is out of range, the input is too short or not finite, or
/// a parameter is out of range.
///
/// # Examples
/// See the [module documentation](self).
pub fn mellin_transform<F: AudioFloat>(
    y: ArrayViewD<'_, F>,
    config: &MellinConfig,
    axis: Axis,
) -> AudioSampleResult<ArrayD<Complex<F>>> {
    if axis.index() >= y.ndim() {
        return invalid("axis", "is out of range");
    }
    let n = y.len_of(axis);
    if n < 3 {
        return invalid("y", format!("{n} samples are fewer than 3"));
    }
    if !(config.t_min.is_finite() && config.t_min > 0.0) {
        return invalid("t_min", "must be a positive number");
    }
    if !(config.over_sample.is_finite() && config.over_sample > 0.0) {
        return invalid("over_sample", "must be positive and finite");
    }
    let nf = n as f64;
    let (n_fmt, log_base) = match config.n_fmt {
        None => {
            if config.over_sample < 1.0 {
                return invalid("over_sample", "must be at least 1");
            }
            let log_base = (nf - 1.0).ln() - (nf - 2.0).ln();
            let count =
                (config.over_sample * ((nf - 1.0).ln() - config.t_min.ln()) / log_base).ceil();
            let count =
                crate::utils::check_derived_len("over_sample", "the Mellin grid", count, 1e10)?;
            (count, log_base)
        }
        Some(c) if c < 3 => return invalid("n_fmt", format!("{c} is below 3")),
        Some(c) => {
            let m = c as f64;
            (c, ((m - 1.0).ln() - (m - 2.0).ln()) / config.over_sample)
        }
    };
    if y.iter().any(|v| !v.is_finite()) {
        return invalid("y", "must be finite");
    }

    let positions: Vec<f64> = (0..n).map(|i| i as f64 / nf).collect();
    let n_over = config.over_sample.ceil() as usize;
    let start = (config.t_min.ln() - nf.ln()) / log_base;
    let total = (n_fmt + n_over) as f64;
    let step = -start / total;
    let mut grid: Vec<f64> = (0..n_fmt)
        .map(|i| (log_base * (start + i as f64 * step)).exp())
        .collect();
    let last = positions[n - 1];
    if grid[0] < config.t_min || grid[n_fmt - 1] > (nf - 1.0) / nf {
        let lo = config.t_min / nf;
        for g in &mut grid {
            *g = g.clamp(lo, last);
        }
    }
    if grid.windows(2).any(|w| w[0] == w[1]) {
        return invalid("n_fmt", "the grid has repeated positions");
    }

    let xi = Array1::from(grid);
    let window: Vec<F> = xi
        .iter()
        .map(|&x| of::<F>(x.powf(config.beta) * nf.sqrt() / n_fmt as f64))
        .collect();
    let resampled = interp1d(
        &positions,
        y,
        xi.view().into_dyn(),
        config.kind,
        F::zero(),
        axis,
    )?;
    let plan = F::with_real_fft_planner(|p| p.plan_fft_forward(n_fmt));
    let mut shape = resampled.shape().to_vec();
    shape[axis.index()] = n_fmt / 2 + 1;
    let mut out = ArrayD::from_elem(IxDyn(&shape), Complex::new(F::zero(), F::zero()));
    let mut input = plan.make_input_vec();
    let mut output = plan.make_output_vec();
    let mut scratch = plan.make_scratch_vec();
    Zip::from(out.lanes_mut(axis))
        .and(resampled.lanes(axis))
        .for_each(|mut o, lane| {
            for ((slot, v), w) in input.iter_mut().zip(lane.iter()).zip(&window) {
                *slot = *v * *w;
            }
            // The buffers come from this plan, so their lengths are right.
            if plan
                .process_with_scratch(&mut input, &mut output, &mut scratch)
                .is_ok()
            {
                for (slot, v) in o.iter_mut().zip(&output) {
                    *slot = Complex::new(v.re, v.im);
                }
            }
        });
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn rejects_bad_parameters() {
        let y = array![1.0, 2.0, 3.0, 4.0].into_dyn();
        let run = |c: MellinConfig| mellin_transform(y.view(), &c, Axis(0));
        assert!(run(MellinConfig::new().with_t_min(0.0)).is_err());
        assert!(run(MellinConfig::new().with_n_fmt(Some(2))).is_err());
        assert!(run(MellinConfig::new().with_over_sample(0.5)).is_err());
        let short = array![1.0, 2.0].into_dyn();
        assert!(mellin_transform(short.view(), &MellinConfig::new(), Axis(0)).is_err());
        let nan = array![1.0, f64::NAN, 2.0].into_dyn();
        assert!(mellin_transform(nan.view(), &MellinConfig::new(), Axis(0)).is_err());
        assert!(mellin_transform(y.view(), &MellinConfig::new(), Axis(1)).is_err());
    }

    #[test]
    fn stretching_keeps_the_scale_magnitude() {
        // Stretching by a factor scales the magnitudes by a constant, so the shape is kept.
        let make = |n: usize, rate: f64| {
            Array1::from_iter((0..n).map(|i| (-(i as f64) / rate).exp())).into_dyn()
        };
        let cfg = MellinConfig::new().with_n_fmt(Some(512));
        let a = mellin_transform(make(2048, 256.0).view(), &cfg, Axis(0)).unwrap();
        let b = mellin_transform(make(2048, 128.0).view(), &cfg, Axis(0)).unwrap();
        let mag = |v: &ArrayD<Complex<f64>>| v.iter().map(|z| z.norm()).collect::<Vec<_>>();
        let (ma, mb) = (mag(&a), mag(&b));
        for k in 1..8 {
            assert!((ma[k] / ma[0] - mb[k] / mb[0]).abs() < 0.1, "{k}");
        }
    }
}
