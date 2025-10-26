use crate::{
    AudioSample, AudioSampleError, AudioSampleResult, AudioSamples, ConvertTo, I24,
    operations::types::OnsetConfig,
};

/// Progress callback function type for beat tracking operations.
///
/// Arguments:
/// - `current`: Current progress value
/// - `total`: Total expected work units
/// - `phase`: Description of current processing phase

#[derive(Clone, Debug)]
pub struct BeatTracker {
    pub tempo_bpm: f64,
    pub beat_times: Vec<f64>,
}

pub type ProgressCallback = dyn Fn(ProgressPhase);

#[non_exhaustive]
#[derive(Clone, Copy, Debug)]
pub enum ProgressPhase {
    Init, // beat tracking init
    Forward(f64),
    Backward(f64),
    Complete,
    OnsetEnvelope,         // onset envelope calculation
    BeatTrackingStart,     // starting beat tracker
    BeatDetectionComplete, // final end
}

impl ProgressPhase {
    /// Returns a fractional progress in the range [0.0, 1.0] across the entire beat detection pipeline.
    ///
    /// This allows downstream consumers (e.g. GUI, CLI progress bars) to display progress
    /// without caring about internal phases or percentage mappings.
    pub fn progress_fraction(&self) -> f64 {
        match self {
            ProgressPhase::OnsetEnvelope => 0.0,
            ProgressPhase::BeatTrackingStart => 0.5,
            ProgressPhase::Init => 0.1,
            ProgressPhase::Forward(frac) => 0.2 + 0.4 * frac.clamp(0.0, 1.0),
            ProgressPhase::Backward(frac) => 0.6 + 0.3 * frac.clamp(0.0, 1.0),
            ProgressPhase::Complete | ProgressPhase::BeatDetectionComplete => 1.0,
        }
    }
}

pub fn track_beats(
    onset_env: &[f64],
    tempo_bpm: f64,
    sample_rate: f64,
    tolerance_seconds: Option<f64>,
    hop_size: Option<usize>,
) -> AudioSampleResult<Vec<f64>> {
    track_beats_with_progress(
        onset_env,
        tempo_bpm,
        sample_rate,
        tolerance_seconds,
        hop_size,
        None,
    )
}

#[inline(always)]
fn find_peak_in_window(onset_env: &[f64], start: isize, end: isize) -> isize {
    let slice = &onset_env[start as usize..end as usize];
    let mut best_val = f64::MIN;
    let mut best_rel = 0;

    for (rel, &val) in slice.iter().enumerate() {
        if val > best_val {
            best_val = val;
            best_rel = rel;
        }
    }

    start + best_rel as isize
}

/// Track beats with optional progress reporting.
pub fn track_beats_with_progress(
    onset_env: &[f64],
    tempo_bpm: f64,
    sample_rate: f64,
    tolerance_seconds: Option<f64>,
    hop_size: Option<usize>,
    progress_callback: Option<&ProgressCallback>,
) -> AudioSampleResult<Vec<f64>> {
    // --- Validation ---
    if onset_env.is_empty() {
        return Err(AudioSampleError::InvalidInput {
            msg: "Onset envelope is empty".to_string(),
        });
    }

    if tempo_bpm <= 0.0 {
        return Err(AudioSampleError::InvalidInput {
            msg: format!("Invalid tempo: {}", tempo_bpm),
        });
    }

    if let Some(callback) = progress_callback {
        callback(ProgressPhase::Init);
    }

    // --- Strongest Peak ---
    let (mut max_val, mut start_idx) = (f64::MIN, 0);
    for (i, &val) in onset_env.iter().enumerate() {
        if val > max_val {
            max_val = val;
            start_idx = i;
        }
    }

    if let Some(callback) = progress_callback {
        callback(ProgressPhase::BeatTrackingStart);
    }

    // --- Precompute constants ---
    let hop = hop_size.unwrap_or(512) as f64;
    let hop_time = hop / sample_rate;
    let ibi = 60.0 / tempo_bpm;
    let ibi_frames = (ibi / hop_time).round() as isize;
    let tol_frames = tolerance_seconds
        .map(|t| (t / hop_time).round() as isize)
        .unwrap_or((ibi_frames as f64 * 0.1).round() as isize);

    let len = onset_env.len() as isize;
    let start = start_idx as isize;

    // --- Work estimate ---
    let forward_beats = ((len - start) / ibi_frames).max(0);
    let backward_beats = (start / ibi_frames).max(0);
    let total_work = (forward_beats + backward_beats).max(1) as f64; // avoid div by 0

    // --- Beat indices ---
    let mut beat_indices: Vec<usize> = vec![start_idx];

    // --- Forward Tracking ---
    if let Some(callback) = progress_callback {
        callback(ProgressPhase::Forward(0.0));
    }
    let mut idx = start;
    let mut beats_processed = 0;
    while idx + ibi_frames < len {
        let target = idx + ibi_frames;
        let window_start = (target - tol_frames).max(0);
        let window_end = (target + tol_frames).min(len);

        let best_idx = find_peak_in_window(onset_env, window_start, window_end);

        beat_indices.push(best_idx as usize);
        idx = best_idx;
        beats_processed += 1;

        if let Some(callback) = progress_callback {
            let frac = beats_processed as f64 / total_work;
            callback(ProgressPhase::Forward(frac));
        }
    }

    // --- Backward Tracking ---
    if let Some(callback) = progress_callback {
        callback(ProgressPhase::Backward(0.0));
    }

    let mut idx = start;
    beats_processed = 0;
    while idx - ibi_frames > 0 {
        let target = idx - ibi_frames;
        let window_start = (target - tol_frames).max(0);
        let window_end = (target + tol_frames).min(len);

        let best_idx = find_peak_in_window(onset_env, window_start, window_end);

        beat_indices.push(best_idx as usize);
        idx = best_idx;
        beats_processed += 1;

        if let Some(callback) = progress_callback {
            let frac = beats_processed as f64 / total_work;
            callback(ProgressPhase::Forward(frac));
        }
    }

    let beats_in_seconds: Vec<f64> = beat_indices
        .into_iter()
        .map(|idx| idx as f64 * hop_time)
        .collect();

    // --- Finalise ---
    if let Some(callback) = progress_callback {
        callback(ProgressPhase::Complete);
    }

    Ok(beats_in_seconds)
}

pub struct BeatConfig {
    pub tempo_bpm: f64,
    pub tolerance: Option<f64>,
    pub onset_config: OnsetConfig,
}

impl BeatConfig {
    /// Create a new BeatConfig with default settings.
    pub fn new(tempo_bpm: f64) -> Self {
        Self {
            tempo_bpm,
            tolerance: None,
            onset_config: OnsetConfig::default(),
        }
    }

    /// Set the tolerance for beat tracking (as a fraction of inter-beat interval).
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = Some(tolerance);
        self
    }

    /// Set the onset detection configuration.
    pub fn with_onset_config(mut self, config: OnsetConfig) -> Self {
        self.onset_config = config;
        self
    }
}

impl<T: AudioSample> AudioSamples<T>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
{
    pub fn detect_beats(
        &self,
        config: &BeatConfig,
        log_compresion: Option<f64>,
    ) -> AudioSampleResult<BeatTracker> {
        self.detect_beats_with_progress(config, log_compresion, None)
    }

    /// Detect beats with optional progress reporting.
    pub fn detect_beats_with_progress(
        &self,
        config: &BeatConfig,
        log_compression: Option<f64>,
        progress_callback: Option<&ProgressCallback>,
    ) -> AudioSampleResult<BeatTracker> {
        if let Some(callback) = progress_callback {
            callback(ProgressPhase::OnsetEnvelope);
        }

        let onset_env = self.onset_strength_envelope_with_progress(
            &config.onset_config,
            log_compression,
            progress_callback,
        )?;

        if let Some(callback) = progress_callback {
            callback(ProgressPhase::BeatTrackingStart);
        }

        let beats = track_beats_with_progress(
            &onset_env,
            config.tempo_bpm,
            self.sample_rate() as f64,
            config.tolerance,
            Some(config.onset_config.hop_size),
            progress_callback,
        )?;

        if let Some(callback) = progress_callback {
            callback(ProgressPhase::BeatDetectionComplete);
        }

        Ok(BeatTracker {
            tempo_bpm: config.tempo_bpm,
            beat_times: beats,
        })
    }
}
