//! Voice Activity Detection (VAD) operations.
//!
//! This module provides fast, frame-based VAD suitable for segmentation and
//! pre-processing. It is designed to avoid allocations beyond the output mask,
//! and prefers contiguous slice access paths for performance.

use crate::operations::traits::AudioVoiceActivityDetection;
use crate::operations::types::{VadChannelPolicy, VadConfig, VadMethod};
use crate::{
    AudioSample, AudioSampleError, AudioSampleResult, AudioSamples, FeatureError, LayoutError,
    ParameterError, RealFloat, to_precision,
};

/// Internal helper describing a frame iteration plan.
#[derive(Debug, Clone, Copy)]
struct FramePlan {
    frame_size: usize,
    hop_size: usize,
    pad_end: bool,
}

impl FramePlan {
    fn frame_starts(self, total_len: usize) -> impl Iterator<Item = usize> {
        let FramePlan {
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
            } else if start + frame_size > total_len {
                return None;
            }

            let current = start;
            start = start.saturating_add(hop_size);
            Some(current)
        })
    }
}

impl<'a, T: AudioSample> AudioVoiceActivityDetection<'a, T> for AudioSamples<'a, T> {
    fn voice_activity_mask<F: RealFloat>(
        &self,
        config: &VadConfig<F>,
    ) -> AudioSampleResult<Vec<bool>> {
        config.validate()?;

        let plan = FramePlan {
            frame_size: config.frame_size,
            hop_size: config.hop_size,
            pad_end: config.pad_end,
        };

        let total_len = self.samples_per_channel();
        if total_len == 0 {
            return Ok(Vec::new());
        }

        let energy_threshold_db_f64: f64 = to_precision(config.energy_threshold_db);
        let energy_threshold_rms: f64 = 10f64.powf(energy_threshold_db_f64 / 20.0);

        // Raw decisions per frame.
        let mut mask = Vec::new();

        match self.as_mono() {
            Some(mono) => {
                let slice = mono.as_slice().ok_or(AudioSampleError::Layout(
                    LayoutError::NonContiguous {
                        operation: "voice activity detection".to_string(),
                        layout_type: "non-contiguous mono array".to_string(),
                    },
                ))?;

                for start in plan.frame_starts(total_len) {
                    let decision = vad_decision_for_slice(
                        slice,
                        start,
                        total_len,
                        config,
                        energy_threshold_rms,
                    )?;
                    mask.push(decision);
                }
            }
            None => {
                let multi = self.as_multi_channel().ok_or(AudioSampleError::Parameter(
                    ParameterError::invalid_value("audio_data", "Audio must be multi-channel"),
                ))?;

                let multi_view = multi.as_view();
                let channels = multi_view.nrows();
                let samples = multi_view.ncols();

                if samples != total_len {
                    return Err(AudioSampleError::Layout(LayoutError::DimensionMismatch {
                        expected_dims: format!("samples_per_channel={total_len}"),
                        actual_dims: format!("ncols={samples}"),
                        operation: "voice activity detection".to_string(),
                    }));
                }

                for start in plan.frame_starts(total_len) {
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
        }

        // Post-processing: smoothing + hangover + min region duration.
        let mut mask = majority_smooth(mask, config.smooth_frames);
        apply_hangover(&mut mask, config.hangover_frames);
        remove_short_runs(&mut mask, true, config.min_speech_frames);
        remove_short_runs(&mut mask, false, config.min_silence_frames);

        Ok(mask)
    }

    fn speech_regions<F: RealFloat>(
        &self,
        config: &VadConfig<F>,
    ) -> AudioSampleResult<Vec<(usize, usize)>> {
        let mask = self.voice_activity_mask(config)?;
        if mask.is_empty() {
            return Ok(Vec::new());
        }

        let plan = FramePlan {
            frame_size: config.frame_size,
            hop_size: config.hop_size,
            pad_end: config.pad_end,
        };

        let total_len = self.samples_per_channel();
        let mut regions = Vec::new();

        let mut in_region = false;
        let mut region_start_sample = 0usize;
        let mut last_frame_end_sample = 0usize;

        for (frame_idx, start) in plan.frame_starts(total_len).enumerate() {
            let is_speech = mask.get(frame_idx).copied().unwrap_or(false);

            let frame_end = start.saturating_add(config.frame_size).min(total_len);
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

fn vad_decision_for_slice<T: AudioSample, F: RealFloat>(
    samples: &[T],
    frame_start: usize,
    total_len: usize,
    config: &VadConfig<F>,
    energy_threshold_rms: f64,
) -> AudioSampleResult<bool> {
    let (rms, zcr) = frame_rms_and_zcr(
        samples,
        frame_start,
        total_len,
        config.frame_size,
        config.pad_end,
    );
    classify(config, rms, zcr, energy_threshold_rms)
}

fn vad_decision_for_multi<T: AudioSample, F: RealFloat>(
    multi: ndarray::ArrayView2<'_, T>,
    channels: usize,
    samples: usize,
    frame_start: usize,
    config: &VadConfig<F>,
    energy_threshold_rms: f64,
) -> AudioSampleResult<bool> {
    let plan_frame_size = config.frame_size;

    match &config.channel_policy {
        VadChannelPolicy::Channel(ch) => {
            if *ch >= channels {
                return Err(AudioSampleError::Parameter(ParameterError::out_of_range(
                    "channel_policy",
                    ch.to_string(),
                    "0".to_string(),
                    (channels.saturating_sub(1)).to_string(),
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
            let (rms, zcr) =
                frame_rms_and_zcr(slice, frame_start, samples, plan_frame_size, config.pad_end);
            classify(config, rms, zcr, energy_threshold_rms)
        }
        VadChannelPolicy::AverageToMono => {
            // Fast path: contiguous ArrayView2.
            if let Some(flat) = multi.as_slice() {
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

            for ch in 0..channels {
                let row = multi.row(ch);
                let (rms, zcr) = if let Some(slice) = row.as_slice() {
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

fn classify<F: RealFloat>(
    config: &VadConfig<F>,
    rms: f64,
    zcr: f64,
    energy_threshold_rms: f64,
) -> AudioSampleResult<bool> {
    match config.method {
        VadMethod::Energy => Ok(rms >= energy_threshold_rms),
        VadMethod::ZeroCrossing => Ok(zcr >= to_precision::<f64, _>(config.zcr_min)
            && zcr <= to_precision::<f64, _>(config.zcr_max)),
        VadMethod::Combined => {
            let zcr_ok = zcr >= to_precision::<f64, _>(config.zcr_min)
                && zcr <= to_precision::<f64, _>(config.zcr_max);
            Ok(rms >= energy_threshold_rms && zcr_ok)
        }
        VadMethod::Spectral => Err(AudioSampleError::Feature(FeatureError::not_enabled(
            "spectral-analysis",
            "VAD spectral mode",
        ))),
    }
}

fn frame_rms_and_zcr<T: AudioSample>(
    samples: &[T],
    frame_start: usize,
    total_len: usize,
    frame_size: usize,
    pad_end: bool,
) -> (f64, f64) {
    let available = total_len.saturating_sub(frame_start);
    if available == 0 || frame_size == 0 {
        return (0.0, 0.0);
    }

    let frame_len = available.min(frame_size);
    let denom_len = if pad_end { frame_size } else { frame_len };

    let mut sum_sq = 0.0f64;
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

    let zcr_denom = (frame_size.saturating_sub(1)).max(1) as f64;
    let zcr = zc as f64 / zcr_denom;

    (rms, zcr)
}

fn frame_rms_and_zcr_avg<T: AudioSample>(
    flat: &[T],
    channels: usize,
    samples: usize,
    frame_start: usize,
    frame_size: usize,
    pad_end: bool,
) -> (f64, f64) {
    let available = samples.saturating_sub(frame_start);
    if available == 0 || frame_size == 0 || channels == 0 {
        return (0.0, 0.0);
    }

    let frame_len = available.min(frame_size);
    let denom_len = if pad_end { frame_size } else { frame_len };

    let mut sum_sq = 0.0f64;
    let mut zc = 0usize;
    let mut prev_sign = 0i8;

    for i in 0..frame_len {
        let col = frame_start + i;
        let mut acc = 0.0f64;
        for ch in 0..channels {
            let idx = ch * samples + col;
            let v: f64 = flat[idx].convert_to();
            acc += v;
        }
        let x = acc / channels as f64;
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

    let zcr_denom = (frame_size.saturating_sub(1)).max(1) as f64;
    let zcr = zc as f64 / zcr_denom;

    (rms, zcr)
}

fn frame_rms_and_zcr_avg_indexed<T: AudioSample>(
    multi: &ndarray::ArrayView2<'_, T>,
    channels: usize,
    samples: usize,
    frame_start: usize,
    frame_size: usize,
    pad_end: bool,
) -> (f64, f64) {
    let available = samples.saturating_sub(frame_start);
    if available == 0 || frame_size == 0 || channels == 0 {
        return (0.0, 0.0);
    }

    let frame_len = available.min(frame_size);
    let denom_len = if pad_end { frame_size } else { frame_len };

    let mut sum_sq = 0.0f64;
    let mut zc = 0usize;
    let mut prev_sign = 0i8;

    for i in 0..frame_len {
        let col = frame_start + i;
        let mut acc = 0.0f64;
        for ch in 0..channels {
            let v: f64 = multi[(ch, col)].convert_to();
            acc += v;
        }
        let x = acc / channels as f64;
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

    let zcr_denom = (frame_size.saturating_sub(1)).max(1) as f64;
    let zcr = zc as f64 / zcr_denom;

    (rms, zcr)
}

fn frame_rms_and_zcr_indexed<T: AudioSample>(
    multi: &ndarray::ArrayView2<'_, T>,
    ch: usize,
    samples: usize,
    frame_start: usize,
    frame_size: usize,
    pad_end: bool,
) -> (f64, f64) {
    let available = samples.saturating_sub(frame_start);
    if available == 0 || frame_size == 0 {
        return (0.0, 0.0);
    }

    let frame_len = available.min(frame_size);
    let denom_len = if pad_end { frame_size } else { frame_len };

    let mut sum_sq = 0.0f64;
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

    let zcr_denom = (frame_size.saturating_sub(1)).max(1) as f64;
    let zcr = zc as f64 / zcr_denom;

    (rms, zcr)
}

fn majority_smooth(mask: Vec<bool>, window: usize) -> Vec<bool> {
    if window <= 1 || mask.is_empty() {
        return mask;
    }

    let w = window;
    let mut out = vec![false; mask.len()];

    let mut sum = 0i32;
    let mut ring = vec![false; w];

    for i in 0..mask.len() {
        let incoming = mask[i];
        let idx = i % w;
        let outgoing = ring[idx];

        ring[idx] = incoming;
        sum += if incoming { 1 } else { 0 };
        sum -= if outgoing { 1 } else { 0 };

        // For the first (w-1) entries, only use the available prefix.
        let denom = (i + 1).min(w) as i32;
        out[i] = sum * 2 >= denom;
    }

    out
}

fn apply_hangover(mask: &mut [bool], hangover_frames: usize) {
    if hangover_frames == 0 || mask.is_empty() {
        return;
    }

    let mut hold = 0usize;
    for v in mask.iter_mut() {
        if *v {
            hold = hangover_frames;
        } else if hold > 0 {
            *v = true;
            hold -= 1;
        }
    }
}

fn remove_short_runs(mask: &mut [bool], value: bool, min_len: usize) {
    if min_len == 0 || mask.is_empty() {
        return;
    }

    let mut i = 0usize;
    while i < mask.len() {
        if mask[i] != value {
            i += 1;
            continue;
        }

        let start = i;
        while i < mask.len() && mask[i] == value {
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
    use super::*;
    use ndarray::Array1;

    #[test]
    fn vad_silence_is_inactive() {
        let audio = AudioSamples::new_mono(Array1::<f32>::zeros(4096), crate::sample_rate!(44100));
        let cfg = VadConfig::<f32> {
            energy_threshold_db: to_precision(-50.0),
            ..VadConfig::energy_only()
        };

        let mask = audio.voice_activity_mask(&cfg).unwrap();
        assert!(!mask.is_empty());
        assert!(mask.iter().all(|&v| !v));
    }

    #[test]
    fn vad_tone_is_active() {
        let n = 4096;
        let sr = 44100.0;
        let freq = 440.0;
        let data: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / sr as f32;
                (2.0 * std::f32::consts::PI * freq as f32 * t).sin() * 0.5
            })
            .collect();
        let audio = AudioSamples::new_mono(Array1::from(data), crate::sample_rate!(44100));

        let cfg = VadConfig::<f32> {
            energy_threshold_db: to_precision(-35.0),
            ..VadConfig::energy_only()
        };

        let mask = audio.voice_activity_mask(&cfg).unwrap();
        assert!(!mask.is_empty());
        assert!(mask.iter().any(|&v| v));
    }

    #[test]
    fn speech_regions_are_non_overlapping() {
        let mut data = vec![0.0f32; 4096];
        for i in 2048..4096 {
            data[i] = 0.2;
        }
        let audio = AudioSamples::new_mono(Array1::from(data), crate::sample_rate!(44100));

        let cfg = VadConfig::<f32> {
            frame_size: 512,
            hop_size: 256,
            energy_threshold_db: to_precision(-45.0),
            ..VadConfig::energy_only()
        };

        let regions = audio.speech_regions(&cfg).unwrap();
        for w in regions.windows(2) {
            assert!(w[0].1 <= w[1].0);
        }
    }
}
