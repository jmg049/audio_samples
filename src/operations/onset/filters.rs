use std::num::NonZeroUsize;

use non_empty_slice::{NonEmptySlice, NonEmptyVec};

use crate::{AudioSampleError, AudioSampleResult, ParameterError};

#[inline]
pub fn median_filter(
    signal: &NonEmptySlice<f64>,
    filter_length: NonZeroUsize,
) -> AudioSampleResult<NonEmptyVec<f64>> {
    if filter_length.get().is_multiple_of(2) {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "filter_length",
            "Must be odd and > 0",
        )));
    }

    if filter_length.get() == 1 {
        return Ok(signal.to_non_empty_vec());
    }

    let half = filter_length.get() / 2;
    let mut out = Vec::with_capacity(signal.len().get());

    for i in 0..signal.len().get() {
        let lo = i.saturating_sub(half);
        let hi = (i + half + 1).min(signal.len().get());
        let mut window = signal[lo..hi].to_vec();
        window.sort_by(|a, b| a.partial_cmp(b).unwrap());

        out.push(window[window.len() / 2]);
    }
    // safety: signal is non-empty and filter_length > 0 ensures out is non-empty
    let out = unsafe { NonEmptyVec::new_unchecked(out) };
    Ok(out)
}

#[inline]
pub fn rectify_inplace(signal: &mut NonEmptySlice<f64>) {
    for v in signal {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
}

#[inline]
pub fn log_compress_inplace(signal: &mut NonEmptySlice<f64>, alpha: f64) {
    for v in signal {
        *v = (1.0 + alpha * *v).ln();
    }
}
