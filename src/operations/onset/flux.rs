use ndarray::Array2;
use non_empty_slice::NonEmptyVec;
use num_complex::Complex;

#[inline]
pub fn energy_flux(mag: &Array2<f64>) -> NonEmptyVec<f64> {
    let (bins, frames) = mag.dim();
    let mut flux = Vec::with_capacity(frames);
    flux.push(0.0);

    for t in 1..frames {
        let mut acc = 0.0;
        for b in 0..bins {
            let diff = mag[[b, t]] - mag[[b, t - 1]];
            if diff > 0.0 {
                acc += diff;
            }
        }
        flux.push(acc);
    }

    // safety: frames > 1 ensures flux is non-empty
    let flux = unsafe { NonEmptyVec::new_unchecked(flux) };
    flux
}

#[inline]
pub fn magnitude_flux(mag: &Array2<f64>) -> NonEmptyVec<f64> {
    let (bins, frames) = mag.dim();
    let mut flux = Vec::with_capacity(frames);
    flux.push(0.0);

    for t in 1..frames {
        let mut acc = 0.0;
        for b in 0..bins {
            let diff = (mag[[b, t]] - mag[[b, t - 1]]).abs();
            acc += diff;
        }
        flux.push(acc);
    }

    // safety:  frames > 1 ensures flux is non-empty
    let flux = unsafe { NonEmptyVec::new_unchecked(flux) };
    flux
}

#[inline]
pub fn complex_flux(spec: &Array2<Complex<f64>>) -> NonEmptyVec<f64> {
    let (bins, frames) = spec.dim();
    let mut flux = Vec::with_capacity(frames);
    flux.push(0.0);

    for t in 1..frames {
        let mut acc = 0.0;
        for b in 0..bins {
            let diff = spec[[b, t]] - spec[[b, t - 1]];
            acc += diff.norm();
        }
        flux.push(acc);
    }
    // safety: frames > 1 ensures flux is non-empty
    let flux = unsafe { NonEmptyVec::new_unchecked(flux) };
    flux
}

#[inline]
pub fn rectified_complex_flux(spec: &Array2<Complex<f64>>) -> NonEmptyVec<f64> {
    let (bins, frames) = spec.dim();
    let mut flux = Vec::with_capacity(frames);
    flux.push(0.0);

    for t in 1..frames {
        let mut acc = 0.0;
        for b in 0..bins {
            let diff = spec[[b, t]].norm() - spec[[b, t - 1]].norm();
            if diff > 0.0 {
                acc += diff;
            }
        }
        flux.push(acc);
    }

    // safety: frames > 1 ensures flux is non-empty
    let flux = unsafe { NonEmptyVec::new_unchecked(flux) };
    flux
}
