use crate::{
    AudioSample, AudioSampleError, AudioSampleResult, AudioSamples, ConvertTo, I24, ParameterError,
    RealFloat, operations::types::OnsetConfig, to_precision,
};

/// Beat tracking results containing tempo and beat timestamps.
#[derive(Clone, Debug)]
pub struct BeatTracker<F: RealFloat> {
    /// Estimated tempo in beats per minute
    pub tempo_bpm: F,
    /// Beat timestamps in seconds
    pub beat_times: Vec<F>,
}

/// Progress callback function type for beat tracking operations.
///
/// Arguments are passed via ProgressPhase enum containing current progress
/// value, total expected work units, and description of current processing phase.
pub type ProgressCallback<F> = dyn Fn(ProgressPhase<F>);

/// Progress phases for beat tracking operations.
#[non_exhaustive]
#[derive(Clone, Copy, Debug)]
pub enum ProgressPhase<F: RealFloat> {
    /// Beat tracking initialization
    Init,
    /// Forward pass with progress fraction
    Forward(F),
    /// Backward pass with progress fraction
    Backward(F),
    /// Beat tracking complete
    Complete,
    /// Onset envelope calculation phase
    OnsetEnvelope,
    /// Starting beat tracker
    BeatTrackingStart,
    /// Final beat detection complete
    BeatDetectionComplete,
}

impl<F: RealFloat> ProgressPhase<F> {
    /// Returns a fractional progress in the range [0.0, 1.0] across the entire beat detection pipeline.
    ///
    /// This allows downstream consumers (e.g. GUI, CLI progress bars) to display progress
    /// without caring about internal phases or percentage mappings.
    pub fn progress_fraction(&self) -> F {
        match self {
            ProgressPhase::OnsetEnvelope => F::zero(),
            ProgressPhase::BeatTrackingStart => to_precision(0.5),
            ProgressPhase::Init => to_precision(0.1),
            ProgressPhase::Forward(frac) => {
                to_precision::<F, _>(0.2)
                    + to_precision::<F, _>(0.4) * frac.clamp(F::zero(), F::one())
            }
            ProgressPhase::Backward(frac) => {
                to_precision::<F, _>(0.6)
                    + to_precision::<F, _>(0.3) * frac.clamp(F::zero(), F::one())
            }
            ProgressPhase::Complete | ProgressPhase::BeatDetectionComplete => F::one(),
        }
    }
}

/// Track beats in an onset envelope using dynamic programming.
///
/// # Arguments
/// * `onset_env` - Onset strength envelope
/// * `tempo_bpm` - Target tempo in beats per minute
/// * `sample_rate` - Sample rate of the audio
/// * `tolerance_seconds` - Beat timing tolerance (default: 0.04 seconds)
/// * `hop_size` - Hop size for onset detection (default: 512)
///
/// # Returns
/// Vector of beat timestamps in seconds
pub fn track_beats<F: RealFloat>(
    onset_env: &[F],
    tempo_bpm: F,
    sample_rate: F,
    tolerance_seconds: Option<F>,
    hop_size: Option<usize>,
) -> AudioSampleResult<Vec<F>> {
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
fn find_peak_in_window<F: RealFloat>(onset_env: &[F], start: isize, end: isize) -> isize {
    let slice = &onset_env[start as usize..end as usize];
    let mut best_val = F::MIN;
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
///
/// # Panics
///
/// Panics if floating point calculations overflow during frame conversion.
pub fn track_beats_with_progress<F: RealFloat>(
    onset_env: &[F],
    tempo_bpm: F,
    sample_rate: F,
    tolerance_seconds: Option<F>,
    hop_size: Option<usize>,
    progress_callback: Option<&ProgressCallback<F>>,
) -> AudioSampleResult<Vec<F>> {
    // --- Validation ---
    if onset_env.is_empty() {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "onset_env",
            "Onset envelope is empty",
        )));
    }

    if tempo_bpm <= F::zero() {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "tempo_bpm",
            format!("Invalid tempo: {}", tempo_bpm),
        )));
    }

    if let Some(callback) = progress_callback {
        callback(ProgressPhase::Init);
    }

    // --- Strongest Peak ---
    let (mut max_val, mut start_idx) = (F::MIN, 0);
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
    let hop = to_precision::<F, _>(hop_size.unwrap_or(512));
    let hop_time: F = hop / sample_rate;
    let ibi: F = to_precision::<F, _>(60.0) / tempo_bpm;
    let ibi_frames = (ibi / hop_time)
        .round()
        .to_isize()
        .expect("Should not fail");
    let tol_frames = tolerance_seconds
        .map(|t| (t / hop_time).round().to_isize().expect("Should not fail"))
        .unwrap_or(
            (to_precision::<F, _>(ibi_frames) * to_precision::<F, _>(0.1))
                .round()
                .to_isize()
                .expect("Should not fail"),
        );

    let len = onset_env.len() as isize;
    let start = start_idx as isize;

    // --- Work estimate ---
    let forward_beats = ((len - start) / ibi_frames).max(0);
    let backward_beats = (start / ibi_frames).max(0);
    let total_work: F = to_precision((forward_beats + backward_beats).max(1)); // avoid div by 0

    // --- Beat indices ---
    let mut beat_indices: Vec<usize> = vec![start_idx];

    // --- Forward Tracking ---
    if let Some(callback) = progress_callback {
        callback(ProgressPhase::Forward(F::zero()));
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
            let frac = to_precision::<F, _>(beats_processed) / total_work;
            callback(ProgressPhase::Forward(frac));
        }
    }

    // --- Backward Tracking ---
    if let Some(callback) = progress_callback {
        callback(ProgressPhase::Backward(F::zero()));
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
            let frac = to_precision::<F, _>(beats_processed) / total_work;
            callback(ProgressPhase::Forward(frac));
        }
    }

    let beats_in_seconds: Vec<F> = beat_indices
        .into_iter()
        .map(|idx| to_precision::<F, _>(idx) * hop_time)
        .collect();

    // --- Finalise ---
    if let Some(callback) = progress_callback {
        callback(ProgressPhase::Complete);
    }

    Ok(beats_in_seconds)
}

/// Configuration for beat tracking operations.
pub struct BeatConfig<F: RealFloat> {
    /// Target tempo in beats per minute
    pub tempo_bpm: F,
    /// Beat timing tolerance in seconds
    pub tolerance: Option<F>,
    /// Configuration for onset detection
    pub onset_config: OnsetConfig<F>,
}

impl<F: RealFloat> BeatConfig<F> {
    /// Create a new BeatConfig with default settings.
    pub fn new(tempo_bpm: F) -> Self {
        Self {
            tempo_bpm,
            tolerance: None,
            onset_config: OnsetConfig::<F>::default(),
        }
    }

    /// Set the tolerance for beat tracking (as a fraction of inter-beat interval).
    pub const fn with_tolerance(mut self, tolerance: F) -> Self {
        self.tolerance = Some(tolerance);
        self
    }

    /// Set the onset detection configuration.
    pub const fn with_onset_config(mut self, config: OnsetConfig<F>) -> Self {
        self.onset_config = config;
        self
    }
}

impl<'a, T: AudioSample> AudioSamples<'a, T>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
{
    /// Detects beat positions in the audio signal.
    ///
    /// # Arguments
    /// * `config` - Beat detection configuration parameters
    /// * `log_compresion` - Optional logarithmic compression factor
    ///
    /// # Returns
    /// A `BeatTracker` containing detected beat positions and tempo information
    pub fn detect_beats<F>(
        &self,
        config: &BeatConfig<F>,
        log_compresion: Option<F>,
    ) -> AudioSampleResult<BeatTracker<F>>
    where
        F: RealFloat + ConvertTo<T>,
        T: ConvertTo<F>,
    {
        self.detect_beats_with_progress(config, log_compresion, None)
    }

    /// Detect beats with optional progress reporting.
    pub fn detect_beats_with_progress<F>(
        &self,
        config: &BeatConfig<F>,
        log_compression: Option<F>,
        progress_callback: Option<&ProgressCallback<F>>,
    ) -> AudioSampleResult<BeatTracker<F>>
    where
        F: RealFloat + ConvertTo<T>,
        T: ConvertTo<F>,
    {
        if let Some(callback) = progress_callback {
            callback(ProgressPhase::OnsetEnvelope);
        }

        let onset_env = self.onset_strength_envelope_with_progress::<F>(
            &config.onset_config,
            log_compression,
            progress_callback,
        )?;

        if let Some(callback) = progress_callback {
            callback(ProgressPhase::BeatTrackingStart);
        }

        let beats: Vec<F> = track_beats_with_progress(
            &onset_env,
            config.tempo_bpm,
            to_precision(self.sample_rate),
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
