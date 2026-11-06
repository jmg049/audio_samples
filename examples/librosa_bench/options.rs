//! Native half of the `options` benchmark group. Rows that use array forms have no native
//! half: they are NumPy and existing kernels.

use audio_samples::operations::constant_q::{
    CqtConfig, VqtConfig, WaveletConfig, griffinlim_cqt_random, vqt_with_intervals,
    wavelet_lengths_with_alpha_into, wavelet_with_alpha,
};
use audio_samples::operations::effects::{
    NnAggregate, SilenceConfig, SparseRecurrence, nn_filter_recurrence_with, nn_filter_sparse,
    split_db_with, trim_db_with,
};
use audio_samples::operations::features::{
    PolyFeaturesConfig, ReassignedSpectrogramConfig, SpectralBandwidthConfig,
    SpectralRolloffConfig, poly_features_from_spectrogram_with_freqs,
    spectral_bandwidth_from_spectrogram_with_freqs, spectral_centroid_from_spectrogram_with_freqs,
    spectral_rolloff_from_spectrogram_with_freqs,
};
use audio_samples::operations::rhythm::{
    FourierTempogramConfig, OnsetStrengthConfig, fourier_tempo_frequencies,
    fourier_tempogram_from_envelope, metrogram_per_bin, onset_strength_multi_slices,
};
use audio_samples::operations::sequence::{
    RecurrenceConfig, RecurrenceMode, SyncAggregate, recurrence_matrix_weighted, sync_slices,
};
use audio_samples::operations::spectrum::{MfccConfig, StftConfig};
use audio_samples::operations::tonal::ChromaStftConfig;
use audio_samples::operations::transforms::magphase_powf;
use audio_samples::utils::array_ops::{chirp, tone, zero_crossings};
use audio_samples::utils::audio_math::note_to_midi_fractional;
use audio_samples::utils::notation::{midi_name, plimit_intervals};
use audio_samples::{
    AudioConstantQ, AudioFeatures, AudioFloat, AudioRhythm, AudioSpectrum, AudioTonal, nzu,
};
use ndarray::{Array2, Array3, Axis, Slice, s};
use std::num::NonZeroUsize;

use crate::{AudioCtx, ConstCtx, SeqCtx, Thunk, VecCtx, thunk};

const SR: f64 = 22050.0;

fn freqs() -> Vec<f64> {
    (0..24)
        .map(|i| 110.0 * 2f64.powf(i as f64 / 12.0))
        .collect()
}

fn alpha() -> Vec<f64> {
    (0..24).map(|i| 0.05 + 0.002 * i as f64).collect()
}

/// `|D|` and the reassigned frequencies of librosa's default reassigned spectrogram with
/// the NaNs filled, of the first channel.
fn reassigned<F: AudioFloat>(x: &AudioCtx<F>) -> (Array3<F>, Array3<F>) {
    let cfg = ReassignedSpectrogramConfig::librosa().with_fill_nan(true);
    let r = x.audio.reassigned_spectrogram(&cfg).unwrap();
    (r.mags, r.freqs)
}

fn mel_slices() -> [Slice; 3] {
    [
        Slice::new(0, Some(32), 1),
        Slice::new(32, Some(96), 2),
        Slice::new(96, None, 1),
    ]
}

fn chroma<F: AudioFloat>(x: &AudioCtx<F>) -> Array2<F> {
    x.audio
        .chroma_stft(&ChromaStftConfig::librosa())
        .unwrap()
        .index_axis_move(Axis(0), 0)
}

/// The median of `v`, as `numpy.median`.
fn median<F: AudioFloat>(v: impl Iterator<Item = F>) -> F {
    let mut v: Vec<F> = v.collect();
    v.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / F::of(2.0)
    }
}

pub fn audio<'a, F: AudioFloat>(case: &str, x: &'a AudioCtx<F>) -> Option<Thunk<'a>> {
    match case {
        "magphase_power" => {
            let c = StftConfig::librosa()
                .with_n_fft(nzu!(512))
                .with_hop_length(NonZeroUsize::new(128));
            let d = x.audio.stft_matrix(&c).unwrap().index_axis_move(Axis(0), 0);
            thunk(move || magphase_powf(d.view(), F::of(1.5)))
        }
        "griffinlim_cqt_random" => {
            let c = CqtConfig::librosa();
            let mag = x.audio.cqt(&c).unwrap().mapv(|z| z.norm());
            let rate = x.rate();
            let n = NonZeroUsize::new(x.n());
            thunk(move || griffinlim_cqt_random::<F>(&mag, &c, rate, 32, 0.99, n, Some(0)).unwrap())
        }
        "centroid_freq2d" => {
            let (s, f) = reassigned(x);
            thunk(move || {
                spectral_centroid_from_spectrogram_with_freqs(s.view(), f.view()).unwrap()
            })
        }
        "bandwidth_freq2d" => {
            let (s, f) = reassigned(x);
            let c = SpectralBandwidthConfig::librosa();
            thunk(move || {
                spectral_bandwidth_from_spectrogram_with_freqs(s.view(), f.view(), None, &c)
                    .unwrap()
            })
        }
        "rolloff_freq2d" => {
            let (s, f) = reassigned(x);
            let c = SpectralRolloffConfig::librosa();
            thunk(move || {
                spectral_rolloff_from_spectrogram_with_freqs(s.view(), f.view(), &c).unwrap()
            })
        }
        "poly_freq2d" => {
            let (s, f) = reassigned(x);
            let c = PolyFeaturesConfig::librosa();
            thunk(move || {
                poly_features_from_spectrogram_with_freqs(s.view(), f.view(), &c).unwrap()
            })
        }
        "onset_multi_slices" => {
            let c = OnsetStrengthConfig::librosa();
            let bands = mel_slices();
            thunk(move || onset_strength_multi_slices(&x.audio, &c, &bands).unwrap())
        }
        "metrogram_none" => {
            let env = x
                .audio
                .onset_strength(&OnsetStrengthConfig::librosa())
                .unwrap();
            let ft =
                fourier_tempogram_from_envelope(env.view(), &FourierTempogramConfig::librosa())
                    .unwrap();
            let n = ft.dim().2 - 1;
            let mag = ft.slice(s![.., .., ..n]).mapv(|z| z.norm());
            let freqs = fourier_tempo_frequencies(f64::from(x.sr), nzu!(512), nzu!(384));
            thunk(move || metrogram_per_bin(mag.view(), &freqs, None, 0.0).unwrap())
        }
        "sync_slices" => {
            let m = x
                .audio
                .mfcc_frames(&MfccConfig::librosa())
                .unwrap()
                .index_axis_move(Axis(0), 0);
            let slices: Vec<Slice> = (0..m.ncols() as isize)
                .step_by(21)
                .map(|i| Slice::new(i, Some(i + 42), 2))
                .collect();
            thunk(move || sync_slices(m.view(), &slices, SyncAggregate::Mean).unwrap())
        }
        "nn_filter_sparse" => {
            // The affinity recurrence as compressed sparse columns.
            let ch = chroma(x);
            let cfg = RecurrenceConfig::librosa().with_mode(RecurrenceMode::Affinity);
            let rec = recurrence_matrix_weighted(ch.view(), &cfg).unwrap();
            let n = rec.ncols();
            let (mut indptr, mut indices, mut values) = (vec![0usize], Vec::new(), Vec::new());
            for i in 0..n {
                for j in 0..n {
                    if rec[[j, i]] != F::zero() {
                        indices.push(j);
                        values.push(rec[[j, i]]);
                    }
                }
                indptr.push(indices.len());
            }
            thunk(move || {
                let r = SparseRecurrence {
                    indptr: &indptr,
                    indices: &indices,
                    values: Some(&values),
                };
                nn_filter_sparse(ch.view(), &r, NnAggregate::WeightedAverage).unwrap()
            })
        }
        "nn_filter_max" => {
            let ch = chroma(x);
            let cfg = RecurrenceConfig::librosa();
            thunk(move || {
                nn_filter_recurrence_with(ch.view(), &cfg, |n, frame: &mut [F]| {
                    for (o, col) in frame.iter_mut().zip(n.columns()) {
                        *o = col.iter().copied().fold(F::neg_infinity(), F::max);
                    }
                })
                .unwrap()
            })
        }
        "trim_ref_callable" => {
            let c = SilenceConfig::new(30.0);
            thunk(move || {
                trim_db_with(
                    &x.audio,
                    &c,
                    |rms| Ok(median(rms.iter().copied()).to_f64().unwrap_or(0.0)),
                    |db| {
                        Ok(db.map_axis(Axis(0), |c| {
                            c.iter().copied().fold(F::neg_infinity(), F::max)
                        }))
                    },
                )
                .unwrap()
            })
        }
        "split_aggregate_callable" => {
            let c = SilenceConfig::new(30.0);
            thunk(move || {
                split_db_with(
                    &x.audio,
                    &c,
                    |rms| {
                        Ok(rms
                            .iter()
                            .copied()
                            .fold(F::neg_infinity(), F::max)
                            .to_f64()
                            .unwrap_or(0.0))
                    },
                    |db| {
                        Ok(if db.nrows() == 1 {
                            db.row(0).to_owned()
                        } else {
                            db.map_axis(Axis(0), |c| median(c.iter().copied()))
                        })
                    },
                )
                .unwrap()
            })
        }
        "vqt_intervals" => {
            let ratios = plimit_intervals(&[3, 5], 12, true).unwrap();
            let cqt = CqtConfig::new()
                .with_fmin(Some(110.0))
                .with_n_bins(NonZeroUsize::new(48))
                .with_tuning(Some(0.0));
            let c = VqtConfig::new().with_cqt(cqt).with_gamma(Some(0.0));
            thunk(move || vqt_with_intervals(&x.audio, &c, &ratios).unwrap())
        }
        _ => None,
    }
}

pub fn constant<'a, F: AudioFloat>(case: &str, _x: &'a ConstCtx<F>) -> Option<Thunk<'a>> {
    match case {
        "tone_phi" => thunk(|| tone::<f64>(440.0, SR, 220_500, 0.3)),
        "chirp_phi" => thunk(|| chirp::<f64>(110.0, 3520.0, SR, 220_500, false, 0.3).unwrap()),
        "midi_to_note_key" => thunk(|| {
            (0..1000)
                .map(|i| {
                    midi_name(
                        48.0 + 48.0 * f64::from(i) / 1000.0,
                        true,
                        true,
                        "F:maj",
                        false,
                    )
                    .unwrap()
                })
                .collect::<Vec<_>>()
        }),
        "hz_to_note_key" => thunk(|| {
            (0..1000)
                .map(|i| {
                    let f = 110.0 * 2f64.powf(f64::from(i) * 3.0 / 1000.0);
                    let midi = 12.0 * (f.log2() - 440f64.log2()) + 69.0;
                    midi_name(midi, false, false, "F:maj", false).unwrap()
                })
                .collect::<Vec<_>>()
        }),
        "note_to_midi_fractional" => thunk(|| {
            ["C4+25", "D4", "E4-10", "G4", "A4", "C5", "F#3", "Bb4"]
                .iter()
                .cycle()
                .take(1000)
                .map(|n| note_to_midi_fractional(n).unwrap())
                .collect::<Vec<_>>()
        }),
        "wavelet_lengths_alpha" => {
            let (f, a) = (freqs(), alpha());
            let c = WaveletConfig::default();
            // librosa's lengths come from NumPy's small-block cache, with no allocation
            // during the call; the native form writes into a buffer allocated once.
            let mut lengths = vec![0.0; f.len()];
            thunk(move || wavelet_lengths_with_alpha_into(&f, SR, &c, &a, &mut lengths).unwrap())
        }
        "wavelet_alpha" => {
            let (f, a) = (freqs(), alpha());
            let c = WaveletConfig::default();
            thunk(move || wavelet_with_alpha(&f, SR, &c, &a).unwrap())
        }
        _ => None,
    }
}

pub fn vec<'a, F: AudioFloat>(case: &str, x: &'a VecCtx<F>) -> Option<Thunk<'a>> {
    match case {
        "zero_crossings_opt" => {
            let v = x.values(0.0, 20.0).mapv(|e| e.sin());
            let t = F::of(0.1);
            thunk(move || zero_crossings(&v, t, true, false, Axis(0)).unwrap())
        }
        _ => None,
    }
}

pub fn seq<'a, F: AudioFloat>(_case: &str, _x: &'a SeqCtx<F>) -> Option<Thunk<'a>> {
    None
}
