use ndarray::Array2;
use non_empty_slice::{NonEmptySlice, NonEmptyVec};

/// Energy-based onset detection function.
#[inline]
pub fn energy_odf(mag: &Array2<f64>) -> NonEmptyVec<f64> {
    let (bins, frames) = mag.dim();
    let mut odf = Vec::with_capacity(frames);
    odf.push(0.0);

    for t in 1..frames {
        let mut acc = 0.0;
        for b in 0..bins {
            let diff = mag[[b, t]] - mag[[b, t - 1]];
            if diff > 0.0 {
                acc += diff;
            }
        }
        odf.push(acc);
    }
    // safety: frames > 1 ensures odf is non-empty
    unsafe { NonEmptyVec::new_unchecked(odf) }
}

/// Apply adaptive threshold in-place.
#[inline]
pub fn apply_adaptive_threshold(
    signal: &mut NonEmptySlice<f64>,
    median: &NonEmptySlice<f64>,
    multiplier: f64,
) {
    debug_assert!(signal.len() == median.len());

    for i in 0..signal.len().get() {
        let thresh = median[i] * multiplier;
        if signal[i] < thresh {
            signal[i] = 0.0;
        }
    }
}
