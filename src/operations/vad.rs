//! Voice Activity Detection (VAD) operations.
//!
//! This module provides fast, frame-based VAD suitable for segmentation and
//! pre-processing. It is designed to avoid allocations beyond the output mask,
//! and prefers contiguous slice access paths for performance.

use std::num::NonZeroUsize;

use non_empty_slice::{NonEmptySlice, NonEmptyVec, non_empty_vec};

use crate::AudioTypeConversion;
use crate::operations::traits::AudioVoiceActivityDetection;
use crate::operations::types::{VadChannelPolicy, VadConfig, VadMethod};
use crate::{
    AudioSampleError, AudioSampleResult, AudioSamples, FeatureError, LayoutError, ParameterError,
    traits::StandardSample,
};

/// Internal helper describing a frame iteration plan.
#[derive(Debug, Clone, Copy)]
struct FramePlan {
    frame_size: NonZeroUsize,
    hop_size: NonZeroUsize,
    pad_end: bool,
}

impl FramePlan {
    fn frame_starts(self, total_len: usize) -> impl Iterator<Item = usize> {
        let Self {
            frame_size,
            hop_size,
            pad_end,
        } = self;

        let mut start = 0usize;
        std::iter::from_fn(move || {
            if pad_end {
                if start >= total_len {
                    return None;
                }
            } else if start + frame_size.get() > total_len {
                return None;
            }

            let current = start;
            start = start.saturating_add(hop_size.get());
            Some(current)
        })
    }
}

impl<T> AudioVoiceActivityDetection for AudioSamples<'_, T>
where
    T: StandardSample,
    Self: AudioTypeConversion<Sample = T>,
{
    fn voice_activity_mask(&self, config: &VadConfig) -> AudioSampleResult<NonEmptyVec<bool>> {
        config.validate()?;

        let plan = FramePlan {
            frame_size: config.frame_size,
            hop_size: config.hop_size,
            pad_end: config.pad_end,
        };

        let total_len = self.samples_per_channel();

        let energy_threshold_rms = 10.0f64.powf(config.energy_threshold_db / 20.0);

        // Raw decisions per frame.
        let mut mask = Vec::new();

        if let Some(mono) = self.as_mono() {
            let slice =
                mono.as_slice()
                    .ok_or(AudioSampleError::Layout(LayoutError::NonContiguous {
                        operation: "voice activity detection".to_string(),
                        layout_type: "non-contiguous mono array".to_string(),
                    }))?;
            // safety: slice is non-empty because total_len > 0.
            let slice = unsafe { NonEmptySlice::new_unchecked(slice) };
            for start in plan.frame_starts(total_len.get()) {
                let decision =
                    vad_decision_for_slice(slice, start, total_len, config, energy_threshold_rms)?;
                mask.push(decision);
            }
        } else {
            let multi = self.as_multi_channel().ok_or(AudioSampleError::Parameter(
                ParameterError::invalid_value("audio_data", "Audio must be multi-channel"),
            ))?;

            let multi_view = multi.as_view();
            let channels = multi_view.nrows();
            let samples = multi_view.ncols();
            // safety: AudioSamples guarantees non-zero channels.
            let channels = unsafe { NonZeroUsize::new_unchecked(channels) };
            // safety: AudioSamples guarantees non-zero samples per channel.
            let samples = unsafe { NonZeroUsize::new_unchecked(samples) };

            if samples != total_len {
                return Err(AudioSampleError::Layout(LayoutError::DimensionMismatch {
                    expected_dims: format!("samples_per_channel={total_len}"),
                    actual_dims: format!("ncols={samples}"),
                    operation: "voice activity detection".to_string(),
                }));
            }

            for start in plan.frame_starts(total_len.get()) {
                let decision = vad_decision_for_multi(
                    multi_view,
                    channels,
                    samples,
                    start,
                    config,
                    energy_threshold_rms,
                )?;
                mask.push(decision);
            }
        }
        // safety: mask is non-empty because total_len > 0 and frame plan yields at least one frame.
        let mask = unsafe { NonEmptyVec::new_unchecked(mask) };
        // Post-processing: smoothing + hangover + min region duration.
        let mut mask = majority_smooth(mask, config.smooth_frames);
        apply_hangover(&mut mask, config.hangover_frames);
        remove_short_runs(&mut mask, true, config.min_speech_frames);
        remove_short_runs(&mut mask, false, config.min_silence_frames);

        Ok(mask)
    }

    fn speech_regions(&self, config: &VadConfig) -> AudioSampleResult<Vec<(usize, usize)>>
where {
        let mask = self.voice_activity_mask(config)?;

        let plan = FramePlan {
            frame_size: config.frame_size,
            hop_size: config.hop_size,
            pad_end: config.pad_end,
        };

        let total_len = self.samples_per_channel().get();
        let mut regions = Vec::new();

        let mut in_region = false;
        let mut region_start_sample = 0usize;
        let mut last_frame_end_sample = 0usize;

        for (frame_idx, start) in plan.frame_starts(total_len).enumerate() {
            let is_speech = mask.get(frame_idx).copied().unwrap_or(false);

            let frame_end = start.saturating_add(config.frame_size.get()).min(total_len);
            last_frame_end_sample = frame_end;

            if is_speech {
                if !in_region {
                    in_region = true;
                    region_start_sample = start;
                }
            } else if in_region {
                regions.push((region_start_sample, frame_end));
                in_region = false;
            }
        }

        if in_region {
            regions.push((region_start_sample, last_frame_end_sample));
        }

        Ok(merge_overlapping_regions(regions))
    }
}

fn vad_decision_for_slice<T>(
    samples: &NonEmptySlice<T>,
    frame_start: usize,
    total_len: NonZeroUsize,
    config: &VadConfig,
    energy_threshold_rms: f64,
) -> AudioSampleResult<bool>
where
    T: StandardSample,
{
    let (rms, zcr) = frame_rms_and_zcr(
        samples,
        frame_start,
        total_len,
        config.frame_size,
        config.pad_end,
    );
    classify(config, rms, zcr, energy_threshold_rms)
}

fn vad_decision_for_multi<T>(
    multi: ndarray::ArrayView2<'_, T>,
    channels: NonZeroUsize,
    samples: NonZeroUsize,
    frame_start: usize,
    config: &VadConfig,
    energy_threshold_rms: f64,
) -> AudioSampleResult<bool>
where
    T: StandardSample,
{
    let plan_frame_size = config.frame_size;

    match &config.channel_policy {
        VadChannelPolicy::Channel(ch) => {
            if *ch >= channels.get() {
                return Err(AudioSampleError::Parameter(ParameterError::out_of_range(
                    "channel_policy",
                    ch.to_string(),
                    "0".to_string(),
                    (channels.get().saturating_sub(1)).to_string(),
                    "channel index out of range".to_string(),
                )));
            }
            let row = multi.row(*ch);
            let slice =
                row.as_slice()
                    .ok_or(AudioSampleError::Layout(LayoutError::NonContiguous {
                        operation: "voice activity detection".to_string(),
                        layout_type: "non-contiguous channel row".to_string(),
                    }))?;
            // safety: slice is non-empty because samples is non-zero.
            let slice = unsafe { NonEmptySlice::new_unchecked(slice) };
            let (rms, zcr) =
                frame_rms_and_zcr(slice, frame_start, samples, plan_frame_size, config.pad_end);
            classify(config, rms, zcr, energy_threshold_rms)
        }
        VadChannelPolicy::AverageToMono => {
            // Fast path: contiguous ArrayView2.
            if let Some(flat) = multi.as_slice() {
                // safety: flat is non-empty because channels and samples are non-zero.
                let flat = unsafe { NonEmptySlice::new_unchecked(flat) };
                let (rms, zcr) = frame_rms_and_zcr_avg(
                    flat,
                    channels,
                    samples,
                    frame_start,
                    plan_frame_size,
                    config.pad_end,
                );
                classify(config, rms, zcr, energy_threshold_rms)
            } else {
                let (rms, zcr) = frame_rms_and_zcr_avg_indexed(
                    &multi,
                    channels,
                    samples,
                    frame_start,
                    plan_frame_size,
                    config.pad_end,
                );
                classify(config, rms, zcr, energy_threshold_rms)
            }
        }
        VadChannelPolicy::AnyChannel | VadChannelPolicy::AllChannels => {
            let want_any = matches!(config.channel_policy, VadChannelPolicy::AnyChannel);
            let mut any_active = false;
            let mut all_active = true;

            for ch in 0..channels.get() {
                let row = multi.row(ch);
                let (rms, zcr) = if let Some(slice) = row.as_slice() {
                    // safety: slice is non-empty because samples is non-zero.
                    let slice = unsafe { NonEmptySlice::new_unchecked(slice) };
                    frame_rms_and_zcr(slice, frame_start, samples, plan_frame_size, config.pad_end)
                } else {
                    frame_rms_and_zcr_indexed(
                        &multi,
                        ch,
                        samples,
                        frame_start,
                        plan_frame_size,
                        config.pad_end,
                    )
                };

                let active = classify(config, rms, zcr, energy_threshold_rms)?;
                any_active |= active;
                all_active &= active;

                if want_any && any_active {
                    return Ok(true);
                }
                if !want_any && !all_active {
                    return Ok(false);
                }
            }

            Ok(if want_any { any_active } else { all_active })
        }
    }
}

fn classify(
    config: &VadConfig,
    rms: f64,
    zcr: f64,
    energy_threshold_rms: f64,
) -> AudioSampleResult<bool> {
    match config.method {
        VadMethod::Energy => Ok(rms >= energy_threshold_rms),
        VadMethod::ZeroCrossing => Ok(zcr >= config.zcr_min && zcr <= config.zcr_max),
        VadMethod::Combined => {
            let zcr_ok = zcr >= config.zcr_min && zcr <= config.zcr_max;
            Ok(rms >= energy_threshold_rms && zcr_ok)
        }
        VadMethod::Spectral => Err(AudioSampleError::Feature(FeatureError::not_enabled(
            "spectral-analysis",
            "VAD spectral mode",
        ))),
    }
}

fn frame_rms_and_zcr<T>(
    samples: &NonEmptySlice<T>,
    frame_start: usize,
    total_len: NonZeroUsize,
    frame_size: NonZeroUsize,
    pad_end: bool,
) -> (f64, f64)
where
    T: StandardSample,
{
    let available = total_len.get().saturating_sub(frame_start);
    if available == 0 {
        return (0.0, 0.0);
    }

    let frame_len = available.min(frame_size.get());
    let denom_len = if pad_end { frame_size.get() } else { frame_len };

    let mut sum_sq: f64 = 0.0;
    let mut zc = 0usize;

    let mut prev_sign = 0i8;

    for i in 0..frame_len {
        let x: f64 = samples[frame_start + i].convert_to();
        sum_sq += x * x;

        let sign = if x > 0.0 {
            1
        } else if x < 0.0 {
            -1
        } else {
            0
        };
        if i > 0 && sign != 0 && prev_sign != 0 && sign != prev_sign {
            zc += 1;
        }
        if sign != 0 {
            prev_sign = sign;
        }
    }

    let rms = if denom_len == 0 {
        0.0
    } else {
        (sum_sq / denom_len as f64).sqrt()
    };

    let zcr_denom = (frame_size.get().saturating_sub(1)).max(1) as f64;
    let zcr = zc as f64 / zcr_denom;

    (rms, zcr)
}

fn frame_rms_and_zcr_avg<T>(
    flat: &NonEmptySlice<T>,
    channels: NonZeroUsize,
    samples: NonZeroUsize,
    frame_start: usize,
    frame_size: NonZeroUsize,
    pad_end: bool,
) -> (f64, f64)
where
    T: StandardSample,
{
    let available = samples.get().saturating_sub(frame_start);
    if available == 0 {
        return (0.0, 0.0);
    }

    let frame_len = available.min(frame_size.get());
    let denom_len = if pad_end { frame_size.get() } else { frame_len };

    let mut sum_sq = 0.0;
    let mut zc = 0usize;
    let mut prev_sign = 0i8;

    for i in 0..frame_len {
        let col = frame_start + i;
        let mut acc = 0.0;
        for ch in 0..channels.get() {
            let idx = ch * samples.get() + col;
            let v: f64 = flat[idx].convert_to();
            acc += v;
        }
        let x = acc / channels.get() as f64;
        sum_sq += x * x;

        let sign = if x > 0.0 {
            1
        } else if x < 0.0 {
            -1
        } else {
            0
        };
        if i > 0 && sign != 0 && prev_sign != 0 && sign != prev_sign {
            zc += 1;
        }
        if sign != 0 {
            prev_sign = sign;
        }
    }

    let rms = if denom_len == 0 {
        0.0
    } else {
        (sum_sq / denom_len as f64).sqrt()
    };

    let zcr_denom = (frame_size.get().saturating_sub(1)).max(1) as f64;
    let zcr = zc as f64 / zcr_denom;
    (rms, zcr)
}

fn frame_rms_and_zcr_avg_indexed<T>(
    multi: &ndarray::ArrayView2<'_, T>,
    channels: NonZeroUsize,
    samples: NonZeroUsize,
    frame_start: usize,
    frame_size: NonZeroUsize,
    pad_end: bool,
) -> (f64, f64)
where
    T: StandardSample,
{
    let available = samples.get().saturating_sub(frame_start);
    if available == 0 {
        return (0.0, 0.0);
    }

    let frame_len = available.min(frame_size.get());
    let denom_len = if pad_end { frame_size.get() } else { frame_len };

    let mut sum_sq = 0.0;
    let mut zc = 0usize;
    let mut prev_sign = 0i8;

    for i in 0..frame_len {
        let col = frame_start + i;
        let mut acc: f64 = 0.0;
        for ch in 0..channels.get() {
            let v: f64 = multi[(ch, col)].convert_to();
            acc += v;
        }
        let x = acc / channels.get() as f64;
        sum_sq += x * x;

        let sign = if x > 0.0 {
            1
        } else if x < 0.0 {
            -1
        } else {
            0
        };
        if i > 0 && sign != 0 && prev_sign != 0 && sign != prev_sign {
            zc += 1;
        }
        if sign != 0 {
            prev_sign = sign;
        }
    }

    let rms = if denom_len == 0 {
        0.0
    } else {
        (sum_sq / denom_len as f64).sqrt()
    };

    let zcr_denom = (frame_size.get().saturating_sub(1)).max(1) as f64;
    let zcr = zc as f64 / zcr_denom;
    (rms, zcr)
}

fn frame_rms_and_zcr_indexed<T>(
    multi: &ndarray::ArrayView2<'_, T>,
    ch: usize,
    samples: NonZeroUsize,
    frame_start: usize,
    frame_size: NonZeroUsize,
    pad_end: bool,
) -> (f64, f64)
where
    T: StandardSample,
{
    let available = samples.get().saturating_sub(frame_start);
    if available == 0 {
        return (0.0, 0.0);
    }

    let frame_len = available.min(frame_size.get());
    let denom_len = if pad_end { frame_size.get() } else { frame_len };

    let mut sum_sq = 0.0;
    let mut zc = 0usize;
    let mut prev_sign = 0i8;

    for i in 0..frame_len {
        let col = frame_start + i;
        let x: f64 = multi[(ch, col)].convert_to();
        sum_sq += x * x;

        let sign = if x > 0.0 {
            1
        } else if x < 0.0 {
            -1
        } else {
            0
        };
        if i > 0 && sign != 0 && prev_sign != 0 && sign != prev_sign {
            zc += 1;
        }
        if sign != 0 {
            prev_sign = sign;
        }
    }

    let rms = if denom_len == 0 {
        0.0
    } else {
        (sum_sq / denom_len as f64).sqrt()
    };

    let zcr_denom = (frame_size.get().saturating_sub(1)).max(1) as f64;
    let zcr = zc as f64 / zcr_denom;

    (rms, zcr)
}

fn majority_smooth(mask: NonEmptyVec<bool>, window: NonZeroUsize) -> NonEmptyVec<bool> {
    if window.get() == 1 {
        return mask;
    }

    let w = window.get();
    let mut out = non_empty_vec![false; mask.len()];

    let mut sum = 0i32;
    let mut ring = vec![false; w];

    for i in 0..mask.len().get() {
        let incoming = mask[i];
        let idx = i % w;
        let outgoing = ring[idx];

        ring[idx] = incoming;
        sum += i32::from(incoming);
        sum -= i32::from(outgoing);

        // For the first (w-1) entries, only use the available prefix.
        let denom = (i + 1).min(w) as i32;
        out[i] = sum * 2 >= denom;
    }

    out
}

fn apply_hangover(mask: &mut NonEmptySlice<bool>, hangover_frames: NonZeroUsize) {
    let mut hold = 0usize;
    for v in mask.iter_mut() {
        if *v {
            hold = hangover_frames.get();
        } else if hold > 0 {
            *v = true;
            hold -= 1;
        }
    }
}

fn remove_short_runs(mask: &mut NonEmptySlice<bool>, value: bool, min_len: usize) {
    let mut i = 0usize;
    while i < mask.len().get() {
        if mask[i] != value {
            i += 1;
            continue;
        }

        let start = i;
        while i < mask.len().get() && mask[i] == value {
            i += 1;
        }
        let end = i;

        if end - start < min_len {
            for v in &mut mask[start..end] {
                *v = !value;
            }
        }
    }
}

fn merge_overlapping_regions(mut regions: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    if regions.len() <= 1 {
        return regions;
    }

    regions.sort_by_key(|(s, _)| *s);

    let mut out = Vec::with_capacity(regions.len());
    let mut current = regions[0];

    for (s, e) in regions.into_iter().skip(1) {
        if s <= current.1 {
            current.1 = current.1.max(e);
        } else {
            out.push(current);
            current = (s, e);
        }
    }

    out.push(current);
    out
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use crate::sample_rate;

    use super::*;
    use ndarray::Array1;

    #[test]
    fn vad_silence_is_inactive() {
        let audio: AudioSamples<'_, f32> =
            AudioSamples::zeros_mono(crate::nzu!(4096), crate::sample_rate!(44100));
        let cfg = VadConfig {
            energy_threshold_db: -50.0,
            ..VadConfig::energy_only()
        };

        let mask = audio.voice_activity_mask(&cfg).unwrap();
        assert!(mask.iter().all(|&v| !v));
    }

    #[test]
    fn vad_tone_is_active() {
        let n = 4096;
        let sr = sample_rate!(44100);
        let freq = 440.0;
        let audio = crate::sine_wave::<f64>(
            freq,
            Duration::from_secs_f64(n as f64 / sr.get() as f64),
            sr,
            0.5,
        );

        let cfg = VadConfig {
            energy_threshold_db: -35.0,
            ..VadConfig::energy_only()
        };

        let mask = audio.voice_activity_mask(&cfg).unwrap();
        assert!(mask.iter().any(|&v| v));
    }

    #[test]
    fn speech_regions_are_non_overlapping() {
        let mut data = vec![0.0f32; 4096];
        for i in 2048..4096 {
            data[i] = 0.2;
        }
        let audio = AudioSamples::new_mono(Array1::from(data), crate::sample_rate!(44100)).unwrap();

        let cfg = VadConfig {
            frame_size: crate::nzu!(512),
            hop_size: crate::nzu!(256),
            energy_threshold_db: -45.0,
            ..VadConfig::energy_only()
        };

        let regions = audio.speech_regions(&cfg).unwrap();
        for w in regions.windows(2) {
            assert!(w[0].1 <= w[1].0);
        }
    }
}
