//! Amplitude envelope extraction for [`AudioSamples`].
//!
//! This module implements the [`AudioEnvelopes`] trait, which provides
//! five methods for tracking how the amplitude of an audio signal
//! varies over time: instantaneous rectification, windowed RMS,
//! moving average, attack/decay following, and the analytic signal
//! envelope.
//!
//! Envelope detection is fundamental to dynamics analysis, onset
//! detection, amplitude modulation, and feature extraction.
//! Gathering these methods behind a single trait gives callers a
//! uniform interface regardless of the underlying sample type
//!
//! Bring [`AudioEnvelopes`] into scope and call methods on any
//! [`AudioSamples`] value.  Each method returns an [`NdResult`] that
//! mirrors the input channel layout: mono audio yields
//! `NdResult::Mono`, multi-channel audio yields
//! `NdResult::MultiChannel`.
//!
//! ```
//! use audio_samples::{AudioSamples, NdResult, sample_rate};
//! use audio_samples::operations::traits::AudioEnvelopes;
//! use ndarray::array;
//!
//! let audio = AudioSamples::new_mono(
//!     array![1.0f32, -0.5, 0.25],
//!     sample_rate!(44100),
//! ).unwrap();
//! let envelope = audio.amplitude_envelope();
//! if let NdResult::Mono(env) = envelope {
//!     assert_eq!(env[0], 1.0f32);
//!     assert_eq!(env[1], 0.5f32); // |-0.5| = 0.5
//! }
//! ```

use std::num::NonZeroUsize;

use ndarray::{Array1, Array2, ArrayView1};
use num_complex::Complex;
use rustfft::FftPlanner;

use crate::operations::dynamic_range::EnvelopeFollower;
use crate::operations::traits::AudioEnvelopes;
use crate::operations::types::DynamicRangeMethod;
use crate::repr::AudioData;
use crate::{AudioSamples, NdResult, StandardSample};

impl<T> AudioEnvelopes for AudioSamples<'_, T>
where
    T: StandardSample,
{
    /// Compute a per-sample rectified amplitude envelope.
    ///
    /// Each output sample is the absolute value of the corresponding
    /// input sample.
    ///
    /// # Returns
    /// An [`NdResult`] matching the input channel layout:
    /// - [`NdResult::Mono`] for mono audio.
    /// - [`NdResult::MultiChannel`] for multi-channel audio.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, NdResult, sample_rate};
    /// use audio_samples::operations::traits::AudioEnvelopes;
    /// use ndarray::array;
    ///
    /// let audio = AudioSamples::new_mono(
    ///     array![1.0f32, -0.5, 0.25],
    ///     sample_rate!(44100),
    /// ).unwrap();
    /// let envelope = audio.amplitude_envelope();
    /// if let NdResult::Mono(env) = envelope {
    ///     assert_eq!(env[0], 1.0f32);
    ///     assert_eq!(env[1], 0.5f32); // |-0.5| = 0.5
    /// }
    /// ```
    #[inline]
    fn amplitude_envelope(&self) -> NdResult<Self::Sample> {
        match &self.data {
            AudioData::Mono(mono) => {
                let array = mono.as_view().mapv(|sample| {
                    let value: f64 = sample.cast_into();
                    T::cast_from(value.abs())
                });
                NdResult::Mono(array)
            }
            AudioData::Multi(multi) => {
                let array = multi.as_view().mapv(|sample| {
                    let value: f64 = sample.cast_into();
                    T::cast_from(value.abs())
                });
                NdResult::MultiChannel(array)
            }
        }
    }

    /// Compute the root-mean-square (RMS) envelope using a sliding window.
    ///
    /// The signal is divided into overlapping windows of `window_size`
    /// samples, advancing by `hop_size` samples between windows.  The
    /// RMS of each window becomes one output sample.  Partial windows
    /// at the end of the signal are included.
    ///
    /// # Arguments
    /// - `window_size` – Number of samples in each analysis window.
    /// - `hop_size` – Number of samples to advance between successive
    ///   windows.
    ///
    /// # Returns
    /// An [`NdResult`] matching the input channel layout, with one
    /// value per window.  The output length is approximately
    /// `ceil(samples / hop_size)`.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, NdResult, sample_rate};
    /// use audio_samples::operations::traits::AudioEnvelopes;
    /// use ndarray::array;
    /// use std::num::NonZeroUsize;
    ///
    /// // A constant ±1 signal has RMS of 1.0 per window
    /// let audio = AudioSamples::new_mono(
    ///     array![1.0f32, -1.0, 1.0, -1.0],
    ///     sample_rate!(44100),
    /// ).unwrap();
    /// let w = NonZeroUsize::new(2).unwrap();
    /// let h = NonZeroUsize::new(2).unwrap();
    /// let envelope = audio.rms_envelope(w, h);
    /// if let NdResult::Mono(env) = envelope {
    ///     assert_eq!(env.len(), 2);
    ///     assert!((env[0] - 1.0).abs() < 1e-6);
    /// }
    /// ```
    #[inline]
    fn rms_envelope(
        &self,
        window_size: NonZeroUsize,
        hop_size: NonZeroUsize,
    ) -> NdResult<Self::Sample> {
        let num_channels = self.num_channels().get() as usize;

        let window_len = window_size.get();
        let hop = hop_size.get();

        match &self.data {
            AudioData::Mono(mono) => {
                let rms = compute_windowed_rms::<T>(mono.as_view(), window_len, hop);
                NdResult::Mono(Array1::from_vec(rms))
            }
            AudioData::Multi(multi) => {
                let rms_per_channel: Vec<Vec<T>> = multi
                    .as_view()
                    .rows()
                    .into_iter()
                    .map(|channel| compute_windowed_rms::<T>(channel, window_len, hop))
                    .collect();

                let window_count = rms_per_channel.first().map(Vec::len).unwrap_or_default();

                let array = Array2::from_shape_fn((num_channels, window_count), |(ch, idx)| {
                    rms_per_channel[ch][idx]
                });

                NdResult::MultiChannel(array)
            }
        }
    }

    /// Track amplitude over time with an envelope follower, separating
    /// attack and decay phases.
    ///
    /// For each sample, the [`EnvelopeFollower`] estimates the current
    /// signal level.  Samples where the level is rising contribute to
    /// the *attack* envelope; samples where it is falling contribute
    /// to the *decay* envelope.  Both output envelopes have the same
    /// length as the input.
    ///
    /// # Arguments
    /// - `follower` – A configured [`EnvelopeFollower`] that controls
    ///   attack and release time constants.
    /// - `method` – Detection strategy used to estimate level:
    ///   peak, RMS, or hybrid.
    ///
    /// # Returns
    /// A tuple `(attack, decay)` of [`NdResult`] values, both matching
    /// the input channel layout and length.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, NdResult, sample_rate};
    /// use audio_samples::operations::traits::AudioEnvelopes;
    /// use audio_samples::operations::dynamic_range::EnvelopeFollower;
    /// use audio_samples::operations::types::DynamicRangeMethod;
    /// use ndarray::array;
    ///
    /// let audio = AudioSamples::new_mono(
    ///     array![0.0f32, 0.0, 0.0],
    ///     sample_rate!(44100),
    /// ).unwrap();
    /// let follower = EnvelopeFollower::new(
    ///     1.0, 10.0, 44100.0, DynamicRangeMethod::Peak,
    /// );
    /// let (attack, _decay) = audio
    ///     .attack_decay_envelope(&follower, DynamicRangeMethod::Peak);
    /// if let NdResult::Mono(env) = attack {
    ///     assert!(env.iter().all(|&v| v.abs() < 1e-6));
    /// }
    /// ```
    #[inline]
    fn attack_decay_envelope(
        &self,
        follower: &EnvelopeFollower,
        method: DynamicRangeMethod,
    ) -> (NdResult<Self::Sample>, NdResult<Self::Sample>) {
        let num_channels = self.num_channels().get() as usize;
        let mut attack_per_channel: Vec<Vec<T>> = vec![Vec::new(); num_channels];
        let mut decay_per_channel: Vec<Vec<T>> = vec![Vec::new(); num_channels];

        match &self.data {
            AudioData::Mono(mono) => {
                let (attack, decay) =
                    compute_attack_decay_channel(mono.as_view(), follower, method);
                attack_per_channel[0] = attack;
                decay_per_channel[0] = decay;
            }
            AudioData::Multi(multi) => {
                for (channel_idx, channel_samples) in multi.as_view().rows().into_iter().enumerate()
                {
                    let (attack, decay) =
                        compute_attack_decay_channel(channel_samples, follower, method);
                    attack_per_channel[channel_idx] = attack;
                    decay_per_channel[channel_idx] = decay;
                }
            }
        }

        let attack_result = if num_channels == 1 {
            NdResult::Mono(Array1::from_vec(attack_per_channel.swap_remove(0)))
        } else {
            let sample_count = attack_per_channel.first().map(Vec::len).unwrap_or_default();
            let array = Array2::from_shape_fn((num_channels, sample_count), |(ch, idx)| {
                attack_per_channel[ch][idx]
            });
            NdResult::MultiChannel(array)
        };

        let decay_result = if num_channels == 1 {
            NdResult::Mono(Array1::from_vec(decay_per_channel.swap_remove(0)))
        } else {
            let sample_count = decay_per_channel.first().map(Vec::len).unwrap_or_default();
            let array = Array2::from_shape_fn((num_channels, sample_count), |(ch, idx)| {
                decay_per_channel[ch][idx]
            });
            NdResult::MultiChannel(array)
        };

        (attack_result, decay_result)
    }

    /// Compute the instantaneous amplitude envelope via the analytic signal.
    ///
    /// The analytic signal is obtained by applying a Hilbert transform
    /// to the input.  The instantaneous amplitude at each sample is the
    /// magnitude of the resulting complex signal
    /// (√(xᵣ² + xᵢ²)).  This produces a smooth envelope that tracks
    /// the true amplitude of modulated or band-limited signals more
    /// accurately than simple rectification.
    ///
    /// # Returns
    /// An [`NdResult`] matching the input channel layout, with one
    /// instantaneous amplitude value per input sample.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, NdResult, sample_rate};
    /// use audio_samples::operations::traits::AudioEnvelopes;
    /// use ndarray::array;
    ///
    /// let audio = AudioSamples::new_mono(
    ///     array![1.0f32, 0.0, -1.0, 0.0],
    ///     sample_rate!(44100),
    /// ).unwrap();
    /// let envelope = audio.analytic_envelope();
    /// assert!(matches!(envelope, NdResult::Mono(_)));
    /// ```
    #[inline]
    fn analytic_envelope(&self) -> NdResult<Self::Sample> {
        let num_channels = self.num_channels().get() as usize;

        match &self.data {
            AudioData::Mono(mono) => {
                let envelope = compute_analytic_envelope_channel(mono.as_view());
                NdResult::Mono(Array1::from_vec(envelope))
            }
            AudioData::Multi(multi) => {
                let mut envelopes = Vec::with_capacity(num_channels);
                for channel in multi.as_view().rows() {
                    envelopes.push(compute_analytic_envelope_channel(channel));
                }

                let sample_count = envelopes.first().map(Vec::len).unwrap_or_default();
                let array = Array2::from_shape_fn((num_channels, sample_count), |(ch, idx)| {
                    envelopes[ch][idx]
                });
                NdResult::MultiChannel(array)
            }
        }
    }

    /// Compute a moving-average envelope over the rectified signal.
    ///
    /// The signal is first rectified (absolute value taken per sample),
    /// then a sliding window mean is applied.  The window advances by
    /// `hop_size` samples between successive outputs.  Partial windows
    /// at the end of the signal are included.
    ///
    /// # Arguments
    /// - `window_size` – Number of samples per averaging window.
    /// - `hop_size` – Number of samples to advance between successive
    ///   windows.
    ///
    /// # Returns
    /// An [`NdResult`] matching the input channel layout, with one
    /// mean value per window.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, NdResult, sample_rate};
    /// use audio_samples::operations::traits::AudioEnvelopes;
    /// use ndarray::array;
    /// use std::num::NonZeroUsize;
    ///
    /// // signal [0, 2, 4, 6]: window means are 1.0 and 5.0
    /// let audio = AudioSamples::new_mono(
    ///     array![0.0f32, 2.0, 4.0, 6.0],
    ///     sample_rate!(44100),
    /// ).unwrap();
    /// let w = NonZeroUsize::new(2).unwrap();
    /// let h = NonZeroUsize::new(2).unwrap();
    /// let envelope = audio.moving_average_envelope(w, h);
    /// if let NdResult::Mono(env) = envelope {
    ///     assert_eq!(env.len(), 2);
    ///     assert!((env[0] - 1.0).abs() < 1e-6);
    ///     assert!((env[1] - 5.0).abs() < 1e-6);
    /// }
    /// ```
    #[inline]
    fn moving_average_envelope(
        &self,
        window_size: NonZeroUsize,
        hop_size: NonZeroUsize,
    ) -> NdResult<Self::Sample> {
        let num_channels = self.num_channels().get() as usize;

        match &self.data {
            AudioData::Mono(mono) => {
                let envelope = compute_moving_average_channel(
                    mono.as_view(),
                    window_size.get(),
                    hop_size.get(),
                );
                NdResult::Mono(Array1::from_vec(envelope))
            }
            AudioData::Multi(multi) => {
                let mut envelopes = Vec::with_capacity(num_channels);
                for channel in multi.as_view().rows() {
                    envelopes.push(compute_moving_average_channel(
                        channel,
                        window_size.get(),
                        hop_size.get(),
                    ));
                }

                let frame_count = envelopes.first().map(Vec::len).unwrap_or_default();
                let array = Array2::from_shape_fn((num_channels, frame_count), |(ch, idx)| {
                    envelopes[ch][idx]
                });
                NdResult::MultiChannel(array)
            }
        }
    }
}

fn compute_attack_decay_channel<T>(
    samples: ArrayView1<'_, T>,
    follower: &EnvelopeFollower,
    method: DynamicRangeMethod,
) -> (Vec<T>, Vec<T>)
where
    T: StandardSample,
{
    let mut follower_instance = follower.clone();
    follower_instance.reset();

    let mut previous_envelope = 0.0;
    let mut attack_value = 0.0;
    let mut decay_value = 0.0;

    let mut attack = Vec::with_capacity(samples.len());
    let mut decay = Vec::with_capacity(samples.len());

    for sample in samples {
        let input: f64 = (*sample).cast_into();
        let envelope = follower_instance.process(input, method);

        if envelope >= previous_envelope {
            attack_value = envelope;
        } else {
            decay_value = envelope;
        }

        attack.push(T::cast_from(attack_value));
        decay.push(T::cast_from(decay_value));
        previous_envelope = envelope;
    }

    (attack, decay)
}

fn compute_windowed_rms<T>(
    samples: ArrayView1<'_, T>,
    window_size: usize,
    hop_size: usize,
) -> Vec<T>
where
    T: StandardSample,
{
    if window_size == 0 || samples.is_empty() {
        return Vec::new();
    }

    let values: Vec<f64> = samples.iter().map(|sample| (*sample).cast_into()).collect();
    let rms_values = apply_window_function(&values, window_size, hop_size, |window| {
        let sum_sq: f64 = window.iter().map(|value| value * value).sum();
        let denom = window.len() as f64;
        if denom == 0.0 {
            0.0
        } else {
            (sum_sq / denom).sqrt()
        }
    });

    rms_values.into_iter().map(T::cast_from).collect()
}

fn compute_moving_average_channel<T>(
    samples: ArrayView1<'_, T>,
    window_size: usize,
    hop_size: usize,
) -> Vec<T>
where
    T: StandardSample,
{
    if window_size == 0 || samples.is_empty() {
        return Vec::new();
    }

    let rectified: Vec<f64> = samples
        .iter()
        .map(|sample| {
            let value: f64 = (*sample).cast_into();
            value.abs()
        })
        .collect();

    let averages = apply_window_function(&rectified, window_size, hop_size, |window| {
        let sum: f64 = window.iter().sum();
        sum / window.len() as f64
    });

    averages.into_iter().map(T::cast_from).collect()
}

fn compute_analytic_envelope_channel<T>(samples: ArrayView1<'_, T>) -> Vec<T>
where
    T: StandardSample,
{
    let values: Vec<f64> = samples.iter().map(|sample| (*sample).cast_into()).collect();

    if values.is_empty() {
        return Vec::new();
    }

    let hilbert = hilbert_transform(&values);

    values
        .iter()
        .zip(hilbert.iter())
        .map(|(&real, &imag)| {
            let magnitude = real.hypot(imag);
            T::cast_from(magnitude)
        })
        .collect()
}

fn apply_window_function<F>(
    values: &[f64],
    window_size: usize,
    hop_size: usize,
    mut stat_fn: F,
) -> Vec<f64>
where
    F: FnMut(&[f64]) -> f64,
{
    if window_size == 0 || hop_size == 0 || values.is_empty() {
        return Vec::new();
    }

    let len = values.len();
    let mut results = Vec::with_capacity(len.div_ceil(hop_size));
    let mut start = 0usize;

    while start < len {
        let end = (start + window_size).min(len);
        let window = &values[start..end];
        if window.is_empty() {
            break;
        }

        results.push(stat_fn(window));
        start += hop_size;
    }

    results
}

fn hilbert_transform(values: &[f64]) -> Vec<f64> {
    let len = values.len();

    if len == 0 {
        return Vec::new();
    }

    if len == 1 {
        return vec![0.0];
    }

    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(len);
    let ifft = planner.plan_fft_inverse(len);

    let mut spectrum: Vec<Complex<f64>> = values
        .iter()
        .map(|&sample| Complex::new(sample, 0.0))
        .collect();

    fft.process(&mut spectrum);
    spectrum[0] *= 1.0;

    if len.is_multiple_of(2) {
        spectrum[len / 2] *= 1.0;
        for bin in spectrum.iter_mut().take(len / 2).skip(1) {
            *bin *= 2.0;
        }
        for bin in spectrum.iter_mut().take(len).skip(len / 2 + 1) {
            *bin = Complex::new(0.0, 0.0);
        }
    } else {
        let upper = len.div_ceil(2);
        for bin in spectrum.iter_mut().take(upper).skip(1) {
            *bin *= 2.0;
        }
        for bin in spectrum.iter_mut().take(len).skip(upper) {
            *bin = Complex::new(0.0, 0.0);
        }
    }

    ifft.process(&mut spectrum);

    let scale = 1.0 / len as f64;
    spectrum.into_iter().map(|value| value.im * scale).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AudioSamples, NdResult, sample_rate};
    use ndarray::{Array1, array};
    use std::f64::consts::PI;
    use std::num::NonZeroUsize;

    #[test]
    fn amplitude_envelope_returns_absolute_values_for_mono() {
        let audio =
            AudioSamples::new_mono(array![1.0f32, -0.5, 0.25], sample_rate!(48_000)).unwrap();

        let result = audio.amplitude_envelope();

        match result {
            NdResult::Mono(env) => {
                let values: Vec<f32> = env.to_vec();
                assert_eq!(values, vec![1.0, 0.5, 0.25]);
            }
            NdResult::MultiChannel(_) => panic!("expected mono envelope"),
        }
    }

    #[test]
    fn rms_envelope_respects_window_and_hop() {
        let audio =
            AudioSamples::new_mono(array![1.0f32, -1.0, 1.0, -1.0], sample_rate!(44_100)).unwrap();

        let result =
            audio.rms_envelope(NonZeroUsize::new(2).unwrap(), NonZeroUsize::new(2).unwrap());

        match result {
            NdResult::Mono(env) => {
                let values: Vec<f32> = env.to_vec();
                assert_eq!(values.len(), 2);
                for value in values {
                    assert!((value - 1.0).abs() < 1e-6);
                }
            }
            NdResult::MultiChannel(_) => panic!("expected mono envelope"),
        }
    }

    #[test]
    fn attack_decay_envelope_zero_signal_is_zero() {
        let audio = AudioSamples::new_mono(array![0.0f32, 0.0, 0.0], sample_rate!(44_100)).unwrap();
        let follower =
            EnvelopeFollower::new(1.0, 10.0, audio.sample_rate_hz(), DynamicRangeMethod::Peak);

        let (attack, decay) = audio.attack_decay_envelope(&follower, DynamicRangeMethod::Peak);

        match attack {
            NdResult::Mono(env) => assert!(env.iter().all(|v| (*v - 0.0).abs() < 1e-6)),
            NdResult::MultiChannel(_) => panic!("expected mono envelope"),
        }

        match decay {
            NdResult::Mono(env) => assert!(env.iter().all(|v| (*v - 0.0).abs() < 1e-6)),
            NdResult::MultiChannel(_) => panic!("expected mono envelope"),
        }
    }

    #[test]
    fn analytic_envelope_tracks_amplitude_modulation() {
        let sample_rate = sample_rate!(48_000);
        let sr = sample_rate.get() as f64;
        let len = 2048;
        let carrier_hz = 440.0;
        let mod_hz = 5.0;

        let signal = Array1::from_iter((0..len).map(|n| {
            let t = n as f64 / sr;
            let envelope = 1.0 + 0.5 * (2.0 * PI * mod_hz * t).cos();
            let sample = envelope * (2.0 * PI * carrier_hz * t).cos();
            sample as f32
        }));

        let audio = AudioSamples::new_mono(signal, sample_rate).unwrap();
        let result = audio.analytic_envelope();

        let expected_envelope: Vec<f32> = (0..len)
            .map(|n| {
                let t = n as f64 / sr;
                (1.0 + 0.5 * (2.0 * PI * mod_hz * t).cos()) as f32
            })
            .collect();

        match result {
            NdResult::Mono(env) => {
                let values = env.to_vec();
                let padding = 256;
                for (idx, (value, target)) in
                    values.iter().zip(expected_envelope.iter()).enumerate()
                {
                    if idx < padding || idx >= values.len().saturating_sub(padding) {
                        continue;
                    }

                    assert!(
                        (value - target).abs() < 7e-2,
                        "Envelope mismatch at index {idx}: got {value}, expected {target}"
                    );
                }
            }
            NdResult::MultiChannel(_) => panic!("expected mono envelope"),
        }
    }

    #[test]
    fn moving_average_envelope_matches_manual_average() {
        let audio =
            AudioSamples::new_mono(array![0.0f32, 2.0, 4.0, 6.0], sample_rate!(44_100)).unwrap();

        let result = audio
            .moving_average_envelope(NonZeroUsize::new(2).unwrap(), NonZeroUsize::new(2).unwrap());

        match result {
            NdResult::Mono(env) => {
                let values = env.to_vec();
                assert_eq!(values.len(), 2);
                assert!((values[0] - 1.0).abs() < 1e-6);
                assert!((values[1] - 5.0).abs() < 1e-6);
            }
            NdResult::MultiChannel(_) => panic!("expected mono envelope"),
        }
    }
}
