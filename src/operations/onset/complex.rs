use super::ComplexOnsetConfig;
use ndarray::{Array2, ArrayView2, Axis, s};
use non_empty_slice::NonEmptyVec;
use num_complex::Complex;
use wide::f64x4;

#[inline]
pub fn magnitude_difference(mag: ArrayView2<f64>) -> Array2<f64> {
    let (bins, frames) = mag.dim();
    let mut out = Array2::zeros((bins, frames));

    assert!(mag.is_standard_layout());
    assert!(out.is_standard_layout());

    for (mut out_row, mag_row) in out.axis_iter_mut(Axis(0)).zip(mag.axis_iter(Axis(0))) {
        let curr = mag_row.slice(s![1..]);
        let prev = mag_row.slice(s![..frames - 1]);
        let mut out_slice = out_row.slice_mut(s![1..]);

        for ((o, &c), &p) in out_slice.iter_mut().zip(curr).zip(prev) {
            *o = c - p;
        }
    }

    out
}

#[inline]
fn wrapped_phase_diff(a: f64, b: f64) -> f64 {
    let mut d = a - b;

    if d > std::f64::consts::PI {
        d -= std::f64::consts::TAU;
    } else if d < -std::f64::consts::PI {
        d += std::f64::consts::TAU;
    }

    d
}

#[inline]
pub fn phase_deviation(
    spec: ArrayView2<Complex<f64>>,
    config: &ComplexOnsetConfig,
    sample_rate: f64,
) -> Array2<f64> {
    let (bins, frames) = spec.dim();
    let mut out = Array2::zeros((bins, frames));

    let hop = config.hop_size.get() as f64;
    let tau = std::f64::consts::TAU;

    for (b, (mut out_row, spec_row)) in out
        .axis_iter_mut(Axis(0))
        .zip(spec.axis_iter(Axis(0)))
        .enumerate()
    {
        let f = config.cqt_config.bin_frequency(b);
        let expected = tau * f * hop / sample_rate;

        let mut prev_phase = spec_row[0].arg();

        for t in 1..frames {
            let phase = spec_row[t].arg();
            let diff = wrapped_phase_diff(phase, prev_phase);
            out_row[t] = (diff - expected).abs();
            prev_phase = phase;
        }
    }

    out
}
#[inline]
pub fn combine_complex_odf(
    mag_diff: &Array2<f64>,
    phase_dev: &Array2<f64>,
    config: &ComplexOnsetConfig,
) -> NonEmptyVec<f64> {
    let (bins, frames) = mag_diff.dim();
    let mut odf = vec![0.0; frames];

    let mag_rect = config.magnitude_rectify;
    let phase_rect = config.phase_rectify;

    for t in 0..frames {
        let mag_col = mag_diff.index_axis(Axis(1), t);
        let phase_col = phase_dev.index_axis(Axis(1), t);

        let mut mag_acc = f64x4::ZERO;
        let mut phase_acc = f64x4::ZERO;

        let mut b = 0;

        if mag_rect && phase_rect {
            while b + 4 <= bins {
                let m_view = mag_col.slice(s![b..b + 4]);
                let p_view = phase_col.slice(s![b..b + 4]);
                let m_slice = m_view.as_slice().unwrap();
                let p_slice = p_view.as_slice().unwrap();
                let m =
                    f64x4::new([m_slice[0], m_slice[1], m_slice[2], m_slice[3]]).max(f64x4::ZERO);
                let p =
                    f64x4::new([p_slice[0], p_slice[1], p_slice[2], p_slice[3]]).max(f64x4::ZERO);
                mag_acc += m;
                phase_acc += p;
                b += 4;
            }
        } else {
            while b + 4 <= bins {
                let m_view = mag_col.slice(s![b..b + 4]);
                let p_view = phase_col.slice(s![b..b + 4]);
                let m_slice = m_view.as_slice().unwrap();
                let p_slice = p_view.as_slice().unwrap();
                let m = f64x4::new([m_slice[0], m_slice[1], m_slice[2], m_slice[3]]).abs();
                let p = f64x4::new([p_slice[0], p_slice[1], p_slice[2], p_slice[3]]).abs();
                mag_acc += m;
                phase_acc += p;
                b += 4;
            }
        }

        let mut mag_sum = mag_acc.reduce_add();
        let mut phase_sum = phase_acc.reduce_add();

        for i in b..bins {
            mag_sum += if mag_rect {
                mag_col[i].max(0.0)
            } else {
                mag_col[i].abs()
            };
            phase_sum += if phase_rect {
                phase_col[i].max(0.0)
            } else {
                phase_col[i].abs()
            };
        }

        let mut v = config.magnitude_weight * mag_sum + config.phase_weight * phase_sum;

        if config.log_compression > 0.0 {
            v = (1.0 + config.log_compression * v).ln();
        }

        odf[t] = v;
    }

    // safety: frames > 0 ensures odf is non-empty
    let odf = unsafe { NonEmptyVec::new_unchecked(odf) };
    odf
}
