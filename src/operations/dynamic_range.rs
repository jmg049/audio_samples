//! Dynamic range processing for audio signals.
//!
//! This module implements dynamic range control processors: compressors, limiters,
//! noise gates, and expanders. Supporting utilities include an [`EnvelopeFollower`]
//! for level detection and a [`LookaheadBuffer`] for anticipatory gain reduction.
//!
//! Raw audio signals often have wide dynamic ranges that are impractical for
//! playback, broadcast, or mixing. Dynamic range processors reduce or expand these
//! differences to achieve consistent loudness, protect against clipping, or enhance
//! perceived punch and clarity.
//!
//! All processors are accessed through the [`AudioDynamicRange`] trait. Behaviour is
//! controlled by [`CompressorConfig`] and [`LimiterConfig`], which provide preset
//! constructors for common use cases. The underlying gain computation uses an
//! [`EnvelopeFollower`] for level tracking and optionally a [`LookaheadBuffer`] to
//! apply gain reductions before peaks arrive.
//!
//! # Example
//!
//! ```
//! use audio_samples::{AudioSamples, AudioDynamicRange, sample_rate};
//! use audio_samples::operations::types::CompressorConfig;
//! use ndarray::Array1;
//!
//! let data = Array1::from_vec(vec![0.1f32, 0.8, 0.2, 0.9, 0.1]);
//! let mut audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
//! let config = CompressorConfig::vocal();
//! audio.apply_compressor(&config).unwrap();
//! ```

use non_empty_slice::NonEmptySlice;

use crate::operations::traits::AudioDynamicRange;
use crate::operations::types::{CompressorConfig, DynamicRangeMethod, KneeType, LimiterConfig};
use crate::repr::AudioData;
use crate::traits::StandardSample;
use crate::utils::audio_math::{
    amplitude_to_db as linear_to_db, db_to_amplitude as db_to_linear, ms_to_samples,
};
use crate::{
    AudioSampleError, AudioSampleResult, AudioSamples, AudioTypeConversion, ConvertTo,
    ParameterError,
};
use std::collections::VecDeque;
use std::num::NonZeroUsize;

/// A first-order RC-style envelope follower with separate attack and release time constants.
///
/// ## Purpose
///
/// Tracks the amplitude envelope of an audio signal over time. Each call to
/// [`process`][EnvelopeFollower::process] returns the current smoothed level, which
/// rises during loud passages (governed by `attack_ms`) and falls during quiet passages
/// (governed by `release_ms`).
///
/// ## Intended Usage
///
/// Used internally by the compressor and limiter implementations to derive a smoothed
/// level estimate from the input signal. It can also be used directly when implementing
/// custom dynamic range processors.
///
/// ## Invariants
///
/// - The envelope value is always non-negative.
/// - Attack and release coefficients are in `[0.0, 1.0]`; a value of `0.0` means
///   instantaneous response with no smoothing.
#[derive(Debug, Clone)]
pub struct EnvelopeFollower {
    /// Current envelope value
    envelope: f64,
    /// Attack coefficient (0.0 to 1.0)
    attack_coeff: f64,
    /// Release coefficient (0.0 to 1.0)
    release_coeff: f64,
    /// Current detector state
    detector_state: f64,
    /// RMS window for RMS detection
    rms_window: VecDeque<f64>,
    /// RMS window size
    rms_window_size: usize,
    /// RMS window sum
    rms_sum: f64,
}

impl EnvelopeFollower {
    /// Creates a new envelope follower with the given time constants.
    ///
    /// # Arguments
    ///
    /// - `attack_ms` – Attack time in milliseconds. Controls how quickly the envelope
    ///   rises when the signal level increases. Use `0.0` for instantaneous response.
    /// - `release_ms` – Release time in milliseconds. Controls how quickly the envelope
    ///   falls when the signal level decreases. Use `0.0` for instantaneous response.
    /// - `sample_rate` – Sample rate in Hz used to convert time constants to
    ///   per-sample coefficients.
    /// - `detection_method` – Determines whether peak, RMS, or hybrid level detection
    ///   is used. For `Rms` and `Hybrid` modes, an internal window of approximately
    ///   10 ms is allocated.
    ///
    /// # Returns
    ///
    /// A new `EnvelopeFollower` with the envelope initialised to `0.0`.
    ///
    /// # Example
    ///
    /// ```
    /// use audio_samples::operations::dynamic_range::EnvelopeFollower;
    /// use audio_samples::operations::types::DynamicRangeMethod;
    ///
    /// let follower = EnvelopeFollower::new(5.0, 50.0, 44100.0, DynamicRangeMethod::Peak);
    /// ```
    #[inline]
    #[must_use]
    pub fn new(
        attack_ms: f64,
        release_ms: f64,
        sample_rate: f64,
        detection_method: DynamicRangeMethod,
    ) -> Self {
        let attack_coeff = if attack_ms > 0.0 {
            (-1.0 / (attack_ms * 0.001 * sample_rate)).exp()
        } else {
            0.0
        };

        let release_coeff = if release_ms > 0.0 {
            (-1.0 / (release_ms * 0.001 * sample_rate)).exp()
        } else {
            0.0
        };

        // RMS window size: 10ms for RMS detection
        let rms_window_size = match detection_method {
            DynamicRangeMethod::Rms | DynamicRangeMethod::Hybrid => {
                (0.01 * sample_rate).max(1.0) as usize
            }
            DynamicRangeMethod::Peak => 1,
        };

        Self {
            envelope: 0.0,
            attack_coeff,
            release_coeff,
            detector_state: 0.0,
            rms_window: VecDeque::with_capacity(rms_window_size),
            rms_window_size,
            rms_sum: 0.0,
        }
    }

    /// Processes a single sample and returns the current envelope level.
    ///
    /// The detector value is computed from `input` according to `detection_method`,
    /// then smoothed using the attack coefficient when rising and the release coefficient
    /// when falling.
    ///
    /// # Arguments
    ///
    /// - `input` – The input sample value. The follower tracks amplitude, so any
    ///   numeric scale is accepted.
    /// - `detection_method` – Detection algorithm to apply. `Peak` uses the absolute
    ///   value; `Rms` computes a windowed root-mean-square; `Hybrid` takes the maximum
    ///   of peak and RMS.
    ///
    /// # Returns
    ///
    /// The smoothed envelope value (always `≥ 0.0`).
    ///
    /// # Example
    ///
    /// ```
    /// use audio_samples::operations::dynamic_range::EnvelopeFollower;
    /// use audio_samples::operations::types::DynamicRangeMethod;
    ///
    /// let mut follower = EnvelopeFollower::new(1.0, 10.0, 44100.0, DynamicRangeMethod::Peak);
    /// let level = follower.process(0.5, DynamicRangeMethod::Peak);
    /// assert!(level > 0.0);
    /// ```
    #[inline]
    pub fn process(&mut self, input: f64, detection_method: DynamicRangeMethod) -> f64 {
        let detector_value = match detection_method {
            DynamicRangeMethod::Peak => input.abs(),
            DynamicRangeMethod::Rms => {
                // Add new sample to RMS window
                let sample_squared: f64 = input * input;
                self.rms_window.push_back(sample_squared);
                self.rms_sum += sample_squared;

                // Remove old samples if window is full
                if self.rms_window.len() > self.rms_window_size
                    && let Some(old_sample) = self.rms_window.pop_front()
                {
                    self.rms_sum -= old_sample;
                }

                // Calculate RMS
                if self.rms_window.is_empty() {
                    0.0
                } else {
                    (self.rms_sum / self.rms_window.len() as f64).sqrt()
                }
            }
            DynamicRangeMethod::Hybrid => {
                // Hybrid: max of peak and RMS
                let peak = input.abs();

                // RMS calculation
                let sample_squared = input * input;
                self.rms_window.push_back(sample_squared);
                self.rms_sum += sample_squared;

                if self.rms_window.len() > self.rms_window_size
                    && let Some(old_sample) = self.rms_window.pop_front()
                {
                    self.rms_sum -= old_sample;
                }

                let rms = if self.rms_window.is_empty() {
                    0.0
                } else {
                    (self.rms_sum / self.rms_window.len() as f64).sqrt()
                };

                peak.max(rms)
            }
        };

        // Apply attack or release based on whether detector is rising or falling
        let coeff = if detector_value > self.detector_state {
            self.attack_coeff
        } else {
            self.release_coeff
        };

        self.detector_state = detector_value;
        self.envelope = coeff * self.envelope + (1.0 - coeff) * detector_value;
        self.envelope
    }

    /// Resets all internal state to zero.
    ///
    /// Clears the envelope value, detector state, and any accumulated RMS window data.
    /// Call this between unrelated audio segments to prevent state leaking across
    /// independent signals.
    ///
    /// # Example
    ///
    /// ```
    /// use audio_samples::operations::dynamic_range::EnvelopeFollower;
    /// use audio_samples::operations::types::DynamicRangeMethod;
    ///
    /// let mut follower = EnvelopeFollower::new(1.0, 10.0, 44100.0, DynamicRangeMethod::Peak);
    /// follower.process(0.8, DynamicRangeMethod::Peak);
    /// follower.reset();
    /// let level = follower.process(0.0, DynamicRangeMethod::Peak);
    /// assert_eq!(level, 0.0);
    /// ```
    #[inline]
    pub fn reset(&mut self) {
        self.envelope = 0.0;
        self.detector_state = 0.0;
        self.rms_window.clear();
        self.rms_sum = 0.0;
    }
}

/// A circular delay buffer that enables lookahead processing in limiters and compressors.
///
/// ## Purpose
///
/// Introduces a fixed sample delay between gain computation and the signal it controls.
/// This allows the processor to begin attenuating the signal *before* a peak arrives,
/// preventing transient clipping without the distortion that zero-latency processing
/// can cause.
///
/// ## Intended Usage
///
/// Used internally by [`apply_limiter`][crate::AudioDynamicRange::apply_limiter] and
/// [`apply_compressor`][crate::AudioDynamicRange::apply_compressor]. Can be used
/// directly when implementing custom lookahead processors.
///
/// ## Invariants
///
/// - The buffer always stores exactly `lookahead_samples` samples.
/// - Until the buffer is full (the first `lookahead_samples` calls to
///   [`process`][LookaheadBuffer::process]), the output is `0.0`.
#[derive(Debug, Clone)]
pub struct LookaheadBuffer {
    /// Circular buffer for samples
    buffer: Vec<f64>,
    /// Write position in buffer
    write_pos: usize,
    /// Read position in buffer
    read_pos: usize,
    /// Buffer size
    size: usize,
    /// Whether buffer is full
    is_full: bool,
}

impl LookaheadBuffer {
    /// Creates a new lookahead buffer for the given delay length.
    ///
    /// # Arguments
    ///
    /// - `lookahead_samples` – Number of samples of delay. A value of `N` means the
    ///   output lags the input by `N` samples.
    ///
    /// # Returns
    ///
    /// A new `LookaheadBuffer` initialised with all zeros.
    ///
    /// # Example
    ///
    /// ```
    /// use audio_samples::operations::dynamic_range::LookaheadBuffer;
    /// use std::num::NonZeroUsize;
    ///
    /// // 10 ms lookahead at 44100 Hz
    /// let buf = LookaheadBuffer::new(NonZeroUsize::new(441).unwrap());
    /// ```
    #[inline]
    #[must_use]
    pub fn new(lookahead_samples: NonZeroUsize) -> Self {
        Self {
            buffer: vec![0.0; lookahead_samples.get()],
            write_pos: 0,
            read_pos: 0,
            size: lookahead_samples.get(),
            is_full: false,
        }
    }

    /// Adds a sample to the buffer and returns the correspondingly delayed output.
    ///
    /// The buffer behaves as a FIFO queue with a fixed length equal to the number of
    /// lookahead samples. The first `N` calls return `0.0` while the buffer fills;
    /// subsequent calls return samples in FIFO order, where `N` is the buffer size.
    ///
    /// # Arguments
    ///
    /// - `input` – The incoming sample to enqueue.
    ///
    /// # Returns
    ///
    /// The delayed sample, or `0.0` if the buffer has not yet been fully filled.
    ///
    /// # Example
    ///
    /// ```
    /// use audio_samples::operations::dynamic_range::LookaheadBuffer;
    /// use std::num::NonZeroUsize;
    ///
    /// let mut buf = LookaheadBuffer::new(NonZeroUsize::new(3).unwrap());
    /// assert_eq!(buf.process(1.0), 0.0); // filling
    /// assert_eq!(buf.process(2.0), 0.0); // filling
    /// assert_eq!(buf.process(3.0), 0.0); // filling
    /// assert_eq!(buf.process(4.0), 1.0); // first delayed output
    /// ```
    #[inline]
    pub fn process(&mut self, input: f64) -> f64 {
        // For a lookahead buffer, we need to return the oldest sample
        // when the buffer is full, otherwise return 0.0

        let output = if self.is_full {
            // Return the oldest sample (at read position)
            self.buffer[self.read_pos]
        } else {
            0.0
        };

        // Write new sample at write position
        self.buffer[self.write_pos] = input;

        // If buffer is full, advance read position
        if self.is_full {
            self.read_pos = (self.read_pos + 1) % self.size;
        }

        // Always advance write position
        self.write_pos = (self.write_pos + 1) % self.size;

        // Check if buffer becomes full (write position caught up to read position)
        if self.write_pos == self.read_pos && !self.is_full {
            self.is_full = true;
        }

        output
    }

    /// Returns the sample currently at the write position without consuming it.
    ///
    /// Returns the value stored at the current write position — the sample that will
    /// next be displaced when [`process`][LookaheadBuffer::process] is called.
    ///
    /// # Returns
    ///
    /// The sample at the write position.
    ///
    /// # Example
    ///
    /// ```
    /// use audio_samples::operations::dynamic_range::LookaheadBuffer;
    /// use std::num::NonZeroUsize;
    ///
    /// let buf = LookaheadBuffer::new(NonZeroUsize::new(4).unwrap());
    /// // Newly created buffer contains all zeros.
    /// assert_eq!(buf.peek(), 0.0);
    /// ```
    #[inline]
    #[must_use]
    pub fn peek(&self) -> f64 {
        self.buffer[self.write_pos]
    }

    /// Resets the buffer to its initial all-zero state.
    ///
    /// Clears all stored samples to `0.0` and resets read/write positions. Call this
    /// between unrelated processing passes to prevent stale samples from affecting output.
    ///
    /// # Example
    ///
    /// ```
    /// use audio_samples::operations::dynamic_range::LookaheadBuffer;
    /// use std::num::NonZeroUsize;
    ///
    /// let mut buf = LookaheadBuffer::new(NonZeroUsize::new(2).unwrap());
    /// buf.process(0.5);
    /// buf.reset();
    /// assert_eq!(buf.peek(), 0.0);
    /// ```
    #[inline]
    pub fn reset(&mut self) {
        self.buffer.fill(0.0);
        self.write_pos = 0;
        self.read_pos = 0;
        self.is_full = false;
    }
}

/// Computes the gain reduction in dB for a given input level and compressor settings.
///
/// Returns the amount (in dB, positive = reduction) that should be subtracted from the
/// input to achieve the configured compression. Returns `0.0` for inputs at or below the
/// threshold.
///
/// # Arguments
///
/// - `input_level_db` – Current signal level in dBFS.
/// - `threshold_db` – Compression threshold in dBFS. Gain reduction begins above this level.
/// - `ratio` – Compression ratio (e.g. `4.0` means 4:1). Must be ≥ 1.0.
/// - `knee_type` – Whether the transition at the threshold is abrupt ([`KneeType::Hard`])
///   or gradual ([`KneeType::Soft`]).
/// - `knee_width_db` – Width of the soft-knee region in dB. Only used when `knee_type`
///   is [`KneeType::Soft`].
///
/// # Returns
///
/// Gain reduction in dB (≥ 0.0). Subtract from the input level to obtain the compressed
/// output level.
///
/// # Example
///
/// ```
/// use audio_samples::operations::dynamic_range::calculate_compression_gain;
/// use audio_samples::operations::types::KneeType;
///
/// // 6 dB overshoot at 4:1 ratio → 4.5 dB gain reduction
/// let reduction = calculate_compression_gain(-6.0, -12.0, 4.0, KneeType::Hard, 2.0);
/// assert!((reduction - 4.5).abs() < 1e-10);
/// ```
#[inline]
#[must_use]
pub fn calculate_compression_gain(
    input_level_db: f64,
    threshold_db: f64,
    ratio: f64,
    knee_type: KneeType,
    knee_width_db: f64,
) -> f64 {
    let overshoot = input_level_db - threshold_db;

    if overshoot <= 0.0 {
        return 0.0;
    }

    match knee_type {
        KneeType::Hard => {
            // Hard knee: abrupt transition
            overshoot - overshoot / ratio
        }
        KneeType::Soft => {
            // Soft knee: smooth transition
            let half_knee = knee_width_db / 2.0;

            if overshoot <= half_knee {
                // In the knee region
                let knee_ratio = overshoot / half_knee;
                let smooth_ratio = ((ratio - 1.0) * knee_ratio).mul_add(knee_ratio, 1.0);
                overshoot - overshoot / smooth_ratio
            } else {
                // Above the knee
                let knee_gain = half_knee - half_knee / ratio;
                knee_gain + (overshoot - half_knee) - (overshoot - half_knee) / ratio
            }
        }
    }
}

/// Computes the gain reduction in dB for a given input level and limiter settings.
///
/// Returns the amount (in dB, positive = reduction) that should be subtracted from the
/// input to enforce the ceiling. Returns `0.0` for inputs at or below the ceiling.
///
/// # Arguments
///
/// - `input_level_db` – Current signal level in dBFS.
/// - `ceiling_db` – Limiting ceiling in dBFS. No output level will exceed this value.
/// - `knee_type` – Whether the transition is abrupt ([`KneeType::Hard`]) or gradual
///   ([`KneeType::Soft`]).
/// - `knee_width_db` – Width of the soft-knee region in dB. Only used when `knee_type`
///   is [`KneeType::Soft`].
///
/// # Returns
///
/// Gain reduction in dB (≥ 0.0). Subtract from the input level to obtain the limited
/// output level.
///
/// # Example
///
/// ```
/// use audio_samples::operations::dynamic_range::calculate_limiting_gain;
/// use audio_samples::operations::types::KneeType;
///
/// // 0.5 dB overshoot → 0.5 dB gain reduction (hard limiting)
/// let reduction = calculate_limiting_gain(-0.5, -1.0, KneeType::Hard, 1.0);
/// assert!((reduction - 0.5).abs() < 1e-10);
/// ```
#[inline]
#[must_use]
pub fn calculate_limiting_gain(
    input_level_db: f64,
    ceiling_db: f64,
    knee_type: KneeType,
    knee_width_db: f64,
) -> f64 {
    let overshoot = input_level_db - ceiling_db;

    if overshoot <= 0.0 {
        return 0.0;
    }

    match knee_type {
        KneeType::Hard => {
            // Hard limiting: complete attenuation of overshoot
            overshoot
        }
        KneeType::Soft => {
            // Soft limiting: smooth transition to full limiting
            let half_knee = knee_width_db / 2.0;

            if overshoot <= half_knee {
                // In the knee region: gradual increase in limiting
                let knee_ratio = overshoot / half_knee;
                overshoot * knee_ratio * knee_ratio
            } else {
                // Above the knee: full limiting
                let knee_gain = half_knee;
                knee_gain + (overshoot - half_knee)
            }
        }
    }
}

impl<T> AudioDynamicRange for AudioSamples<'_, T>
where
    T: StandardSample,
    Self: AudioTypeConversion<Sample = T>,
{
    /// Applies compression to reduce the dynamic range of the signal.
    ///
    /// Samples above the threshold are attenuated according to the compression ratio.
    /// Attack and release times control how quickly the compressor responds to level
    /// changes. An optional lookahead delay allows the compressor to anticipate peaks
    /// before they occur. Multi-channel audio is compressed independently per channel.
    ///
    /// # Arguments
    ///
    /// - `config` – Compressor parameters including threshold, ratio, attack, release,
    ///   knee type, makeup gain, lookahead, and detection method. Use preset constructors
    ///   such as [`CompressorConfig::vocal`] or [`CompressorConfig::drum`] for common
    ///   configurations.
    ///
    /// # Returns
    ///
    /// `Ok(())` on success. The audio is modified in place.
    ///
    /// # Errors
    ///
    /// Returns [crate::AudioSampleError::Parameter] if the configuration fails validation
    /// (e.g. threshold above 0 dBFS, ratio below 1.0, or negative time constants).
    ///
    /// # Example
    ///
    /// ```
    /// use audio_samples::{AudioSamples, AudioDynamicRange, sample_rate};
    /// use audio_samples::operations::types::CompressorConfig;
    /// use ndarray::Array1;
    ///
    /// let data = Array1::from_vec(vec![0.1f32, 0.8, 0.2, 0.9, 0.1]);
    /// let mut audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
    /// let config = CompressorConfig::vocal();
    /// audio.apply_compressor(&config).unwrap();
    /// ```
    fn apply_compressor(&mut self, config: &CompressorConfig) -> AudioSampleResult<()> {
        let sample_rate = self.sample_rate_hz();
        // Validate configuration
        config.validate(sample_rate)?;

        // Calculate lookahead buffer size
        let lookahead_samples = ms_to_samples(config.lookahead_ms, sample_rate);
        let lookahead_samples = lookahead_samples.max(1);
        // safety: we ensured lookahead_samples is at least 1, so this unwrap is safe
        let lookahead_samples = unsafe { NonZeroUsize::new_unchecked(lookahead_samples) };
        match &mut self.data {
            AudioData::Mono(samples) => {
                let mut envelope_follower = EnvelopeFollower::new(
                    config.attack_ms,
                    config.release_ms,
                    sample_rate,
                    config.detection_method,
                );

                let mut lookahead_buffer = LookaheadBuffer::new(lookahead_samples);
                let mut gain_reductions = Vec::with_capacity(samples.len().get());

                // Process each sample
                for &sample in samples.iter() {
                    let sample_f: f64 = sample.convert_to();

                    // Get envelope level
                    let envelope = envelope_follower.process(sample_f, config.detection_method);
                    let envelope_db = linear_to_db(envelope);

                    // Calculate gain reduction
                    let gain_reduction_db = calculate_compression_gain(
                        envelope_db,
                        config.threshold_db,
                        config.ratio,
                        config.knee_type,
                        config.knee_width_db,
                    );

                    gain_reductions.push(gain_reduction_db);
                }

                // Apply gain reductions with lookahead
                for i in 0..samples.len().get() {
                    let sample_f: f64 = samples[i].convert_to();
                    let delayed_sample = lookahead_buffer.process(sample_f);

                    if i >= lookahead_samples.get() {
                        let gain_reduction_db = gain_reductions[i - lookahead_samples.get()];
                        let gain_linear = db_to_linear(-gain_reduction_db);
                        let makeup_gain = db_to_linear(config.makeup_gain_db);

                        let output_sample = delayed_sample * gain_linear * makeup_gain;
                        samples[i - lookahead_samples.get()] = output_sample.convert_to();
                    }
                }

                // Process remaining samples in lookahead buffer
                for i in 0..lookahead_samples.get() {
                    let delayed_sample = lookahead_buffer.process(0.0);

                    // Check for overflow before subtraction
                    if lookahead_samples <= samples.len() {
                        let sample_idx = samples.len().get() - lookahead_samples.get() + i;

                        if sample_idx < samples.len().get() {
                            let gain_reduction_db = gain_reductions[sample_idx];
                            let gain_linear = db_to_linear(-gain_reduction_db);
                            let makeup_gain = db_to_linear(config.makeup_gain_db);

                            let output_sample = delayed_sample * gain_linear * makeup_gain;
                            samples[sample_idx] = output_sample.convert_to();
                        }
                    }
                }
            }
            AudioData::Multi(samples) => {
                let num_channels = samples.nrows().get();
                let num_samples = samples.ncols().get();

                // Process each channel independently
                for channel in 0..num_channels {
                    let mut envelope_follower = EnvelopeFollower::new(
                        config.attack_ms,
                        config.release_ms,
                        sample_rate,
                        config.detection_method,
                    );

                    let mut lookahead_buffer = LookaheadBuffer::new(lookahead_samples);
                    let mut gain_reductions = Vec::with_capacity(num_samples);

                    // Process each sample
                    for sample_idx in 0..num_samples {
                        let sample_f: f64 = samples[[channel, sample_idx]].convert_to();

                        // Get envelope level
                        let envelope = envelope_follower.process(sample_f, config.detection_method);
                        let envelope_db = linear_to_db(envelope);

                        // Calculate gain reduction
                        let gain_reduction_db = calculate_compression_gain(
                            envelope_db,
                            config.threshold_db,
                            config.ratio,
                            config.knee_type,
                            config.knee_width_db,
                        );

                        gain_reductions.push(gain_reduction_db);
                    }

                    // Apply gain reductions with lookahead
                    for sample_idx in 0..num_samples {
                        let sample_f: f64 = samples[[channel, sample_idx]].convert_to();
                        let delayed_sample = lookahead_buffer.process(sample_f);

                        if sample_idx >= lookahead_samples.get() {
                            let gain_reduction_db =
                                gain_reductions[sample_idx - lookahead_samples.get()];
                            let gain_linear = db_to_linear(-gain_reduction_db);
                            let makeup_gain = db_to_linear(config.makeup_gain_db);

                            let output_sample = delayed_sample * gain_linear * makeup_gain;
                            samples[[channel, sample_idx - lookahead_samples.get()]] =
                                output_sample.convert_to();
                        }
                    }

                    // Process remaining samples in lookahead buffer
                    for i in 0..lookahead_samples.get() {
                        let delayed_sample = lookahead_buffer.process(0.0);

                        // Check for overflow before subtraction
                        if lookahead_samples.get() <= num_samples {
                            let sample_idx = num_samples - lookahead_samples.get() + i;

                            if sample_idx < num_samples {
                                let gain_reduction_db = gain_reductions[sample_idx];
                                let gain_linear = db_to_linear(-gain_reduction_db);
                                let makeup_gain = db_to_linear(config.makeup_gain_db);

                                let output_sample = delayed_sample * gain_linear * makeup_gain;
                                samples[[channel, sample_idx]] = output_sample.convert_to();
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Prevents the signal from exceeding a specified ceiling level.
    ///
    /// Applies gain reduction to any samples that breach the ceiling. Lookahead
    /// processing delays the output signal while analysing future samples, allowing gain
    /// reduction to begin before peaks arrive and minimising audible distortion.
    /// Multi-channel audio is limited independently per channel.
    ///
    /// # Arguments
    ///
    /// - `config` – Limiter parameters including ceiling, attack, release, knee type,
    ///   lookahead, and detection method. Use preset constructors such as
    ///   [`LimiterConfig::transparent`] or [`LimiterConfig::mastering`] for common
    ///   configurations.
    ///
    /// # Returns
    ///
    /// `Ok(())` on success. The audio is modified in place.
    ///
    /// # Errors
    ///
    /// Returns [crate::AudioSampleError::Parameter] if the configuration fails validation
    /// (e.g. ceiling above 0 dBFS or invalid time constants).
    ///
    /// # Example
    ///
    /// ```
    /// use audio_samples::{AudioSamples, AudioDynamicRange, sample_rate};
    /// use audio_samples::operations::types::LimiterConfig;
    /// use ndarray::Array1;
    ///
    /// let data = Array1::from_vec(vec![0.1f32, 0.8, 0.2, 0.9, 0.1]);
    /// let mut audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
    /// let config = LimiterConfig::mastering();
    /// audio.apply_limiter(&config).unwrap();
    /// ```
    fn apply_limiter(&mut self, config: &LimiterConfig) -> AudioSampleResult<()> {
        let sample_rate = self.sample_rate_hz();
        // Validate configuration
        let config = config.validate(sample_rate)?;

        // Calculate lookahead buffer size
        let lookahead_samples = ms_to_samples(config.lookahead_ms, sample_rate);
        let lookahead_samples = lookahead_samples.max(1);
        // safety: we ensured lookahead_samples is at least 1, so this unwrap is safe
        let lookahead_samples = unsafe { NonZeroUsize::new_unchecked(lookahead_samples) };
        match &mut self.data {
            AudioData::Mono(samples) => {
                let mut envelope_follower = EnvelopeFollower::new(
                    config.attack_ms,
                    config.release_ms,
                    sample_rate,
                    config.detection_method,
                );

                let mut lookahead_buffer = LookaheadBuffer::new(lookahead_samples);
                let mut gain_reductions = Vec::with_capacity(samples.len().get());

                // Process each sample
                for &sample in samples.iter() {
                    let sample_f: f64 = sample.convert_to();

                    // Get envelope level
                    let envelope = envelope_follower.process(sample_f, config.detection_method);
                    let envelope_db = linear_to_db(envelope);

                    // Calculate gain reduction
                    let gain_reduction_db = calculate_limiting_gain(
                        envelope_db,
                        config.ceiling_db,
                        config.knee_type,
                        config.knee_width_db,
                    );

                    gain_reductions.push(gain_reduction_db);
                }

                // Apply gain reductions with lookahead
                for i in 0..samples.len().get() {
                    let sample_f: f64 = samples[i].convert_to();
                    let delayed_sample = lookahead_buffer.process(sample_f);

                    if i >= lookahead_samples.get() {
                        let gain_reduction_db = gain_reductions[i - lookahead_samples.get()];
                        let gain_linear = db_to_linear(-gain_reduction_db);

                        let output_sample = delayed_sample * gain_linear;
                        samples[i - lookahead_samples.get()] = output_sample.convert_to();
                    }
                }

                // Process remaining samples in lookahead buffer
                for i in 0..lookahead_samples.get() {
                    let delayed_sample = lookahead_buffer.process(0.0);

                    // Check for overflow before subtraction
                    if lookahead_samples <= samples.len() {
                        let sample_idx = samples.len().get() - lookahead_samples.get() + i;

                        if sample_idx < samples.len().get() {
                            let gain_reduction_db = gain_reductions[sample_idx];
                            let gain_linear = db_to_linear(-gain_reduction_db);

                            let output_sample = delayed_sample * gain_linear;
                            samples[sample_idx] = output_sample.convert_to();
                        }
                    }
                }
            }
            AudioData::Multi(samples) => {
                let num_channels = samples.nrows().get();
                let num_samples = samples.ncols().get();

                // Process each channel independently
                for channel in 0..num_channels {
                    let mut envelope_follower = EnvelopeFollower::new(
                        config.attack_ms,
                        config.release_ms,
                        sample_rate,
                        config.detection_method,
                    );

                    let mut lookahead_buffer = LookaheadBuffer::new(lookahead_samples);
                    let mut gain_reductions = Vec::with_capacity(num_samples);

                    // Process each sample
                    for sample_idx in 0..num_samples {
                        let sample_f: f64 = samples[[channel, sample_idx]].convert_to();

                        // Get envelope level
                        let envelope = envelope_follower.process(sample_f, config.detection_method);
                        let envelope_db = linear_to_db(envelope);

                        // Calculate gain reduction
                        let gain_reduction_db = calculate_limiting_gain(
                            envelope_db,
                            config.ceiling_db,
                            config.knee_type,
                            config.knee_width_db,
                        );

                        gain_reductions.push(gain_reduction_db);
                    }

                    // Apply gain reductions with lookahead
                    for sample_idx in 0..num_samples {
                        let sample_f: f64 = samples[[channel, sample_idx]].convert_to();
                        let delayed_sample = lookahead_buffer.process(sample_f);

                        if sample_idx >= lookahead_samples.get() {
                            let gain_reduction_db =
                                gain_reductions[sample_idx - lookahead_samples.get()];
                            let gain_linear = db_to_linear(-gain_reduction_db);

                            let output_sample = delayed_sample * gain_linear;
                            samples[[channel, sample_idx - lookahead_samples.get()]] =
                                output_sample.convert_to();
                        }
                    }

                    // Process remaining samples in lookahead buffer
                    for i in 0..lookahead_samples.get() {
                        let delayed_sample = lookahead_buffer.process(0.0);

                        // Check for overflow before subtraction
                        if lookahead_samples.get() <= num_samples {
                            let sample_idx = num_samples - lookahead_samples.get() + i;

                            if sample_idx < num_samples {
                                let gain_reduction_db = gain_reductions[sample_idx];
                                let gain_linear = db_to_linear(-gain_reduction_db);

                                let output_sample = delayed_sample * gain_linear;
                                samples[[channel, sample_idx]] = output_sample.convert_to();
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Applies compression driven by an external sidechain signal.
    ///
    /// The gain reduction is determined by the level of `sidechain_signal` rather than
    /// the main audio, allowing one signal to control the dynamics of another. A common
    /// use case is ducking: a voice track causes background music to decrease in level
    /// whenever speech is present.
    ///
    /// Only mono-to-mono sidechain processing is currently supported. Multi-channel
    /// combinations return an error.
    ///
    /// # Arguments
    ///
    /// - `config` – Compressor configuration. Sidechain processing must be enabled;
    ///   call `config.side_chain.enable()` before passing the config to this method.
    /// - `sidechain_signal` – External control signal. Must be the same length as the
    ///   main signal.
    ///
    /// # Returns
    ///
    /// `Ok(())` on success. The main audio is modified in place.
    ///
    /// # Errors
    ///
    /// - [crate::AudioSampleError::Parameter] if sidechain is not enabled in `config`.
    /// - [crate::AudioSampleError::Parameter] if the main and sidechain signals have
    ///   different lengths.
    /// - [crate::AudioSampleError::Parameter] if either signal is multi-channel (not yet
    ///   supported).
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use audio_samples::{AudioSamples, AudioDynamicRange, sample_rate};
    /// use audio_samples::operations::types::CompressorConfig;
    /// use ndarray::Array1;
    ///
    /// let mut audio = AudioSamples::new_mono(
    ///     Array1::from_vec(vec![0.1f32, 0.8, 0.2, 0.9, 0.1]),
    ///     sample_rate!(44100),
    /// ).unwrap();
    /// let sidechain = AudioSamples::new_mono(
    ///     Array1::from_vec(vec![0.0f32, 1.0, 0.0, 1.0, 0.0]),
    ///     sample_rate!(44100),
    /// ).unwrap();
    /// let mut config = CompressorConfig::new();
    /// config.side_chain.enable();
    /// audio.apply_compressor_sidechain(&config, &sidechain).unwrap();
    /// ```
    fn apply_compressor_sidechain(
        &mut self,
        config: &CompressorConfig,
        sidechain_signal: &Self,
    ) -> AudioSampleResult<()> {
        let sample_rate = self.sample_rate_hz();
        // Validate configuration
        config.validate(sample_rate)?;

        if !config.side_chain.enabled {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "side_chain_config",
                "Side-chain is not enabled in configuration",
            )));
        }

        // For now, implement basic side-chain by using the sidechain signal for detection
        // but applying gain reduction to the main signal
        // This is a simplified implementation - a full implementation would include
        // filtering of the sidechain signal according to the configuration

        match (&mut self.data, &sidechain_signal.data) {
            (AudioData::Mono(main_samples), AudioData::Mono(sc_samples)) => {
                if main_samples.len() != sc_samples.len() {
                    return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                        "signal_lengths",
                        "Main signal and sidechain signal must have the same length",
                    )));
                }

                let mut envelope_follower = EnvelopeFollower::new(
                    config.attack_ms,
                    config.release_ms,
                    sample_rate,
                    config.detection_method,
                );

                let lookahead_samples = ms_to_samples(config.lookahead_ms, sample_rate);
                let lookahead_samples = lookahead_samples.max(1);

                // safety: we ensured lookahead_samples is at least 1, so this unwrap is safe
                let lookahead_samples = unsafe { NonZeroUsize::new_unchecked(lookahead_samples) };
                let mut lookahead_buffer = LookaheadBuffer::new(lookahead_samples);
                let mut gain_reductions = Vec::with_capacity(main_samples.len().get());

                // Process sidechain signal to get gain reductions
                for &sc_sample in sc_samples {
                    let sc_f: f64 = sc_sample.convert_to();

                    // Mix between internal and external sidechain
                    let envelope = envelope_follower.process(sc_f, config.detection_method);
                    let envelope_db = linear_to_db(envelope);

                    // Calculate gain reduction
                    let gain_reduction_db = calculate_compression_gain(
                        envelope_db,
                        config.threshold_db,
                        config.ratio,
                        config.knee_type,
                        config.knee_width_db,
                    );

                    gain_reductions.push(gain_reduction_db);
                }

                // Apply gain reductions to main signal
                for i in 0..main_samples.len().get() {
                    let sample_f: f64 = main_samples[i].convert_to();
                    let delayed_sample = lookahead_buffer.process(sample_f);

                    if i >= lookahead_samples.get() {
                        let gain_reduction_db = gain_reductions[i - lookahead_samples.get()];
                        let gain_linear = db_to_linear(-gain_reduction_db);
                        let makeup_gain = db_to_linear(config.makeup_gain_db);

                        let output_sample = delayed_sample * gain_linear * makeup_gain;
                        main_samples[i - lookahead_samples.get()] = output_sample.convert_to();
                    }
                }

                // Process remaining samples in lookahead buffer
                for i in 0..lookahead_samples.get() {
                    let delayed_sample = lookahead_buffer.process(0.0);
                    let sample_idx = main_samples.len().get() - lookahead_samples.get() + i;

                    if sample_idx < main_samples.len().get() {
                        let gain_reduction_db = gain_reductions[sample_idx];
                        let gain_linear = db_to_linear(-gain_reduction_db);
                        let makeup_gain = db_to_linear(config.makeup_gain_db);

                        let output_sample = delayed_sample * gain_linear * makeup_gain;
                        main_samples[sample_idx] = output_sample.convert_to();
                    }
                }
            }
            _ => {
                return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                    "audio_format",
                    "Side-chain processing with multi-channel audio not yet implemented",
                )));
            }
        }

        Ok(())
    }

    /// Applies limiting driven by an external sidechain signal.
    ///
    /// The gain reduction ceiling is enforced based on the level of `sidechain_signal`
    /// rather than the main audio. Useful for frequency-selective limiting where a
    /// filtered copy of the signal controls gain reduction.
    ///
    /// Only mono-to-mono sidechain processing is currently supported. Multi-channel
    /// combinations return an error.
    ///
    /// # Arguments
    ///
    /// - `config` – Limiter configuration. Sidechain processing must be enabled;
    ///   call `config.side_chain.enable()` before passing the config to this method.
    /// - `sidechain_signal` – External control signal. Must be the same length as the
    ///   main signal.
    ///
    /// # Returns
    ///
    /// `Ok(())` on success. The main audio is modified in place.
    ///
    /// # Errors
    ///
    /// - [crate::AudioSampleError::Parameter] if sidechain is not enabled in `config`.
    /// - [crate::AudioSampleError::Parameter] if the main and sidechain signals have
    ///   different lengths.
    /// - [crate::AudioSampleError::Parameter] if either signal is multi-channel (not yet
    ///   supported).
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use audio_samples::{AudioSamples, AudioDynamicRange, sample_rate};
    /// use audio_samples::operations::types::LimiterConfig;
    /// use ndarray::Array1;
    ///
    /// let mut audio = AudioSamples::new_mono(
    ///     Array1::from_vec(vec![0.1f32, 0.8, 0.2, 0.9, 0.1]),
    ///     sample_rate!(44100),
    /// ).unwrap();
    /// let sidechain = AudioSamples::new_mono(
    ///     Array1::from_vec(vec![0.0f32, 1.0, 0.0, 1.0, 0.0]),
    ///     sample_rate!(44100),
    /// ).unwrap();
    /// let mut config = LimiterConfig::default();
    /// config.side_chain.enable();
    /// audio.apply_limiter_sidechain(&config, &sidechain).unwrap();
    /// ```
    fn apply_limiter_sidechain(
        &mut self,
        config: &LimiterConfig,
        sidechain_signal: &Self,
    ) -> AudioSampleResult<()> {
        let sample_rate = self.sample_rate_hz();
        // Validate configuration
        config.validate(sample_rate)?;
        if !config.side_chain.enabled {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "side_chain_config",
                "Side-chain is not enabled in configuration",
            )));
        }

        // Similar to compressor sidechain but with limiting gain calculation
        match (&mut self.data, &sidechain_signal.data) {
            (AudioData::Mono(main_samples), AudioData::Mono(sc_samples)) => {
                if main_samples.len() != sc_samples.len() {
                    return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                        "signal_lengths",
                        "Main signal and sidechain signal must have the same length",
                    )));
                }

                let mut envelope_follower = EnvelopeFollower::new(
                    config.attack_ms,
                    config.release_ms,
                    sample_rate,
                    config.detection_method,
                );

                let lookahead_samples = ms_to_samples(config.lookahead_ms, sample_rate);
                let lookahead_samples = lookahead_samples.max(1);
                // safety: we ensured lookahead_samples is at least 1
                let lookahead_samples = unsafe { NonZeroUsize::new_unchecked(lookahead_samples) };

                let mut lookahead_buffer = LookaheadBuffer::new(lookahead_samples);
                let mut gain_reductions = Vec::with_capacity(main_samples.len().get());

                // Process sidechain signal to get gain reductions
                for &sc_sample in sc_samples {
                    let sc_f: f64 = sc_sample.convert_to();

                    let envelope = envelope_follower.process(sc_f, config.detection_method);
                    let envelope_db = linear_to_db(envelope);

                    // Calculate gain reduction
                    let gain_reduction_db = calculate_limiting_gain(
                        envelope_db,
                        config.ceiling_db,
                        config.knee_type,
                        config.knee_width_db,
                    );

                    gain_reductions.push(gain_reduction_db);
                }

                // Apply gain reductions to main signal
                for i in 0..main_samples.len().get() {
                    let sample_f: f64 = main_samples[i].convert_to();
                    let delayed_sample = lookahead_buffer.process(sample_f);

                    if i >= lookahead_samples.get() {
                        let gain_reduction_db = gain_reductions[i - lookahead_samples.get()];
                        let gain_linear = db_to_linear(-gain_reduction_db);

                        let output_sample = delayed_sample * gain_linear;
                        main_samples[i - lookahead_samples.get()] = output_sample.convert_to();
                    }
                }

                // Process remaining samples in lookahead buffer
                for i in 0..lookahead_samples.get() {
                    let delayed_sample = lookahead_buffer.process(0.0);
                    let sample_idx = main_samples.len().get() - lookahead_samples.get() + i;

                    if sample_idx < main_samples.len().get() {
                        let gain_reduction_db = gain_reductions[sample_idx];
                        let gain_linear = db_to_linear(-gain_reduction_db);

                        let output_sample = delayed_sample * gain_linear;
                        main_samples[sample_idx] = output_sample.convert_to();
                    }
                }
            }
            _ => {
                return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                    "audio_format",
                    "Side-chain processing with multi-channel audio not yet implemented",
                )));
            }
        }

        Ok(())
    }

    /// Computes the static compression input-output curve for given input levels.
    ///
    /// Maps each input level in `input_levels_db` through the compressor's static gain
    /// characteristic (threshold, ratio, knee) plus makeup gain, returning the resulting
    /// output level in dBFS for each input. Does not use time-varying envelope following —
    /// the result depends only on the static transfer function.
    ///
    /// Useful for visualising and verifying compressor behaviour without processing actual
    /// audio samples.
    ///
    /// # Arguments
    ///
    /// - `config` – Compressor configuration parameters.
    /// - `input_levels_db` – Non-empty slice of input levels in dBFS to evaluate.
    ///
    /// # Returns
    ///
    /// A `Vec<f64>` of output levels in dBFS, one per entry in `input_levels_db`,
    /// in the same order.
    ///
    /// # Example
    ///
    /// ```
    /// use audio_samples::{AudioSamples, AudioDynamicRange, sample_rate};
    /// use audio_samples::operations::types::CompressorConfig;
    /// use non_empty_slice::NonEmptySlice;
    /// use ndarray::Array1;
    ///
    /// let audio = AudioSamples::new_mono(
    ///     Array1::from_elem(10, 0.5f32), sample_rate!(44100),
    /// ).unwrap();
    /// let config = CompressorConfig::new();
    /// let levels = [-40.0f64, -20.0, -12.0, 0.0];
    /// let curve = audio.get_compression_curve(
    ///     &config,
    ///     NonEmptySlice::new(&levels).unwrap(),
    /// ).unwrap();
    /// assert_eq!(curve.len(), 4);
    /// // Levels above the threshold are reduced (output < input)
    /// assert!(curve[3] < 0.0);
    /// ```
    fn get_compression_curve(
        &self,
        config: &CompressorConfig,
        input_levels_db: &NonEmptySlice<f64>,
    ) -> AudioSampleResult<Vec<f64>> {
        let output_levels = input_levels_db
            .iter()
            .map(|&input_db| {
                let gain_reduction_db = calculate_compression_gain(
                    input_db,
                    config.threshold_db,
                    config.ratio,
                    config.knee_type,
                    config.knee_width_db,
                );
                input_db - gain_reduction_db + config.makeup_gain_db
            })
            .collect();

        Ok(output_levels)
    }

    /// Returns the per-sample gain reduction that would be applied by the compressor.
    ///
    /// Passes the audio through the envelope follower and compression gain calculation,
    /// collecting the gain reduction (in dB) at every sample without modifying the
    /// signal. For multi-channel audio, only the first channel is analysed.
    ///
    /// Useful for metering, visualising, and analysing compressor activity.
    ///
    /// # Arguments
    ///
    /// - `config` – Compressor configuration parameters.
    ///
    /// # Returns
    ///
    /// A `Vec<f64>` of gain reduction values in dB (each value ≥ 0.0), one per sample
    /// in the signal (or first channel for multi-channel audio).
    ///
    /// # Example
    ///
    /// ```
    /// use audio_samples::{AudioSamples, AudioDynamicRange, sample_rate};
    /// use audio_samples::operations::types::CompressorConfig;
    /// use ndarray::Array1;
    ///
    /// let data = Array1::from_vec(vec![0.1f32, 0.8, 0.2, 0.9, 0.1]);
    /// let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
    /// let config = CompressorConfig::new();
    /// let reductions = audio.get_gain_reduction(&config).unwrap();
    /// assert_eq!(reductions.len(), 5);
    /// assert!(reductions.iter().all(|&r| r >= 0.0));
    /// ```
    fn get_gain_reduction(&self, config: &CompressorConfig) -> AudioSampleResult<Vec<f64>> {
        let sample_rate = self.sample_rate_hz();

        match &self.data {
            AudioData::Mono(samples) => {
                let mut envelope_follower = EnvelopeFollower::new(
                    config.attack_ms,
                    config.release_ms,
                    sample_rate,
                    config.detection_method,
                );

                let mut gain_reductions = Vec::with_capacity(samples.len().get());

                for &sample in samples {
                    let sample_f: f64 = sample.convert_to();
                    let envelope = envelope_follower.process(sample_f, config.detection_method);
                    let envelope_db = linear_to_db(envelope);

                    let gain_reduction_db = calculate_compression_gain(
                        envelope_db,
                        config.threshold_db,
                        config.ratio,
                        config.knee_type,
                        config.knee_width_db,
                    );

                    gain_reductions.push(gain_reduction_db);
                }

                Ok(gain_reductions)
            }
            AudioData::Multi(samples) => {
                // For multi-channel, return gain reduction for first channel
                let num_samples = samples.ncols().get();
                let mut envelope_follower = EnvelopeFollower::new(
                    config.attack_ms,
                    config.release_ms,
                    sample_rate,
                    config.detection_method,
                );

                let mut gain_reductions = Vec::with_capacity(num_samples);

                for sample_idx in 0..num_samples {
                    let sample_f: f64 = samples[[0, sample_idx]].convert_to();
                    let envelope = envelope_follower.process(sample_f, config.detection_method);
                    let envelope_db = linear_to_db(envelope);

                    let gain_reduction_db = calculate_compression_gain(
                        envelope_db,
                        config.threshold_db,
                        config.ratio,
                        config.knee_type,
                        config.knee_width_db,
                    );

                    gain_reductions.push(gain_reduction_db);
                }

                Ok(gain_reductions)
            }
        }
    }

    /// Attenuates the signal when it falls below a threshold (noise gate).
    ///
    /// A gate mutes or reduces quiet passages — typically background noise, room tone,
    /// or bleed — while leaving louder content unaffected. Signals below `threshold_db`
    /// are attenuated by the given ratio; signals above are passed through unchanged.
    /// Peak detection is always used for gate processing.
    ///
    /// # Arguments
    ///
    /// - `threshold_db` – Gate threshold in dBFS. Signals below this level are attenuated.
    /// - `ratio` – Attenuation ratio below the threshold. Higher values produce more
    ///   aggressive gating; values near 1.0 approach unity gain.
    /// - `attack_ms` – Attack time in milliseconds. Controls how quickly the gate opens
    ///   when the signal rises above the threshold.
    /// - `release_ms` – Release time in milliseconds. Controls how quickly the gate
    ///   closes when the signal falls below the threshold.
    ///
    /// # Returns
    ///
    /// `Ok(())` on success. The audio is modified in place.
    ///
    /// # Errors
    ///
    /// Currently always returns `Ok`. Future versions may validate parameter ranges.
    ///
    /// # Example
    ///
    /// ```
    /// use audio_samples::{AudioSamples, AudioDynamicRange, sample_rate};
    /// use ndarray::Array1;
    ///
    /// let data = Array1::from_vec(vec![0.001f32, 0.8, 0.002, 0.9, 0.001]);
    /// let mut audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
    /// // Gate at -20 dBFS with 10:1 ratio
    /// audio.apply_gate(-20.0, 10.0, 1.0, 10.0).unwrap();
    /// ```
    fn apply_gate(
        &mut self,
        threshold_db: f64,
        ratio: f64,
        attack_ms: f64,
        release_ms: f64,
    ) -> AudioSampleResult<()> {
        let sample_rate = self.sample_rate_hz();
        // Gate is essentially a compressor with inverted threshold logic
        // and very high ratio for signals below threshold

        match &mut self.data {
            AudioData::Mono(samples) => {
                let mut envelope_follower = EnvelopeFollower::new(
                    attack_ms,
                    release_ms,
                    sample_rate,
                    DynamicRangeMethod::Peak, // Use peak detection for gates
                );

                for sample in samples.iter_mut() {
                    let sample_f: f64 = (*sample).convert_to();
                    let envelope = envelope_follower.process(sample_f, DynamicRangeMethod::Peak);
                    let envelope_db = linear_to_db(envelope);

                    // Gate logic: attenuate if below threshold
                    let gain_reduction_db = if envelope_db < threshold_db {
                        let undershoot = threshold_db - envelope_db;
                        undershoot * (ratio - 1.0) / ratio
                    } else {
                        0.0
                    };

                    let gain_linear = db_to_linear(-gain_reduction_db);
                    let output_sample = sample_f * gain_linear;

                    *sample = output_sample.convert_to();
                }
            }
            AudioData::Multi(samples) => {
                let num_channels = samples.nrows().get();
                let num_samples = samples.ncols().get();

                for channel in 0..num_channels {
                    let mut envelope_follower = EnvelopeFollower::new(
                        attack_ms,
                        release_ms,
                        sample_rate,
                        DynamicRangeMethod::Peak,
                    );

                    for sample_idx in 0..num_samples {
                        let sample_f: f64 = samples[[channel, sample_idx]].convert_to();
                        let envelope =
                            envelope_follower.process(sample_f, DynamicRangeMethod::Peak);
                        let envelope_db = linear_to_db(envelope);

                        // Gate logic: attenuate if below threshold
                        let gain_reduction_db = if envelope_db < threshold_db {
                            let undershoot = threshold_db - envelope_db;
                            undershoot * (ratio - 1.0) / ratio
                        } else {
                            0.0
                        };

                        let gain_linear = db_to_linear(-gain_reduction_db);
                        let output_sample = sample_f * gain_linear;

                        samples[[channel, sample_idx]] = output_sample.convert_to();
                    }
                }
            }
        }

        Ok(())
    }

    /// Increases dynamic range by expanding signals below a threshold.
    ///
    /// An expander is the complement of compression: signals below `threshold_db` are
    /// attenuated by an amount that grows with distance from the threshold, making quiet
    /// passages quieter while leaving loud passages unchanged. RMS detection is always
    /// used for expansion.
    ///
    /// # Arguments
    ///
    /// - `threshold_db` – Expansion threshold in dBFS. Signals below this level are
    ///   attenuated.
    /// - `ratio` – Expansion ratio. Values greater than `1.0` produce increasing
    ///   attenuation the further the signal falls below the threshold.
    /// - `attack_ms` – Attack time in milliseconds.
    /// - `release_ms` – Release time in milliseconds.
    ///
    /// # Returns
    ///
    /// `Ok(())` on success. The audio is modified in place.
    ///
    /// # Errors
    ///
    /// Currently always returns `Ok`. Future versions may validate parameter ranges.
    ///
    /// # Example
    ///
    /// ```
    /// use audio_samples::{AudioSamples, AudioDynamicRange, sample_rate};
    /// use ndarray::Array1;
    ///
    /// let data = Array1::from_vec(vec![0.1f32, 0.8, 0.2, 0.9, 0.1]);
    /// let mut audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
    /// // Expand at -20 dBFS with 2:1 ratio
    /// audio.apply_expander(-20.0, 2.0, 1.0, 10.0).unwrap();
    /// ```
    fn apply_expander(
        &mut self,
        threshold_db: f64,
        ratio: f64,
        attack_ms: f64,
        release_ms: f64,
    ) -> AudioSampleResult<()> {
        let sample_rate = self.sample_rate_hz();
        // Expander increases dynamic range by expanding signals below threshold

        match &mut self.data {
            AudioData::Mono(samples) => {
                let mut envelope_follower = EnvelopeFollower::new(
                    attack_ms,
                    release_ms,
                    sample_rate,
                    DynamicRangeMethod::Rms, // Use RMS detection for expanders
                );

                for sample in samples.iter_mut() {
                    let sample_f: f64 = (*sample).convert_to();
                    let envelope = envelope_follower.process(sample_f, DynamicRangeMethod::Rms);
                    let envelope_db = linear_to_db(envelope);

                    // Expander logic: amplify the difference below threshold
                    let gain_change_db = if envelope_db < threshold_db {
                        let undershoot = threshold_db - envelope_db;
                        undershoot * (ratio - 1.0) // Expand by ratio
                    } else {
                        1.0
                    };

                    let gain_linear = db_to_linear(-gain_change_db); // Negative because we're reducing level
                    let output_sample = sample_f * gain_linear;

                    *sample = output_sample.convert_to();
                }
            }
            AudioData::Multi(samples) => {
                let num_channels = samples.nrows().get();
                let num_samples = samples.ncols().get();

                for channel in 0..num_channels {
                    let mut envelope_follower = EnvelopeFollower::new(
                        attack_ms,
                        release_ms,
                        sample_rate,
                        DynamicRangeMethod::Rms,
                    );

                    for sample_idx in 0..num_samples {
                        let sample_f: f64 = samples[[channel, sample_idx]].convert_to();
                        let envelope = envelope_follower.process(sample_f, DynamicRangeMethod::Rms);
                        let envelope_db = linear_to_db(envelope);

                        // Expander logic: amplify the difference below threshold
                        let gain_change_db = if envelope_db < threshold_db {
                            let undershoot = threshold_db - envelope_db;
                            undershoot * (ratio - 1.0) // Expand by ratio
                        } else {
                            0.0
                        };

                        let gain_linear = db_to_linear(-gain_change_db); // Negative because we're reducing level
                        let output_sample = sample_f * gain_linear;

                        samples[[channel, sample_idx]] = output_sample.convert_to();
                    }
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AudioSamples;
    use crate::sample_rate;
    use ndarray::Array1;
    use non_empty_slice::non_empty_vec;

    #[test]
    fn test_compressor_basic() {
        let data = Array1::from_vec(vec![0.1f32, 0.8, 0.2, 0.9, 0.1]);
        let mut audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();

        let config = CompressorConfig::new();
        let result = audio.apply_compressor(&config);

        assert!(result.is_ok());
    }

    #[test]
    fn test_limiter_basic() {
        let data = Array1::from_vec(vec![0.1f32, 0.8, 0.2, 0.9, 0.1]);
        let mut audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();

        let config = LimiterConfig::default();
        let result = audio.apply_limiter(&config);

        assert!(result.is_ok());
    }

    #[test]
    fn test_compression_curve() {
        let data = Array1::from_vec(vec![0.1f32, 0.2, 0.3, 0.4, 0.5]);
        let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();

        let config = CompressorConfig::default();
        let input_levels = non_empty_vec![-40.0, -30.0, -20.0, -10.0, 0.0];
        let result = audio.get_compression_curve(&config, input_levels.as_non_empty_slice());

        assert!(result.is_ok());
        let output_levels = result.unwrap();
        assert_eq!(output_levels.len(), input_levels.len().get());

        // Check that compression reduces output levels above threshold
        for (i, &input_db) in input_levels.iter().enumerate() {
            if input_db > config.threshold_db {
                assert!(
                    output_levels[i] < input_db,
                    "Output level {} should be less than input level {} for compression",
                    output_levels[i],
                    input_db
                );
            }
        }
    }

    #[test]
    fn test_gain_reduction() {
        let data = Array1::from_vec(vec![0.1f32, 0.8, 0.2, 0.9, 0.1]);
        let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();

        let config = CompressorConfig::new();
        let result = audio.get_gain_reduction(&config);

        assert!(result.is_ok());
        let gain_reductions = result.unwrap();
        assert_eq!(gain_reductions.len(), 5);

        // All gain reductions should be non-negative
        for &gr in &gain_reductions {
            assert!(gr >= 0.0, "Gain reduction should be non-negative: {}", gr);
        }
    }

    #[test]
    fn test_gate_basic() {
        let data = Array1::from_vec(vec![0.001f32, 0.8, 0.002, 0.9, 0.001]);
        let mut audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();

        let result = audio.apply_gate(-20.0, 10.0, 1.0, 10.0);

        assert!(result.is_ok());
    }

    #[test]
    fn test_expander_basic() {
        let data = Array1::from_vec(vec![0.1f32, 0.8, 0.2, 0.9, 0.1]);
        let mut audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();

        let result = audio.apply_expander(-20.0, 2.0, 1.0, 10.0);

        assert!(result.is_ok());
    }

    #[test]
    fn test_envelope_follower() {
        let mut envelope = EnvelopeFollower::new(1.0, 10.0, 44100.0, DynamicRangeMethod::Peak);

        // Test that envelope responds to input
        let output1 = envelope.process(0.5, DynamicRangeMethod::Peak);
        let output2 = envelope.process(0.5, DynamicRangeMethod::Peak);

        assert!(output1 > 0.0);
        assert!(output2 > output1); // Should be increasing on attack
    }

    #[test]
    fn test_lookahead_buffer() {
        let mut buffer = LookaheadBuffer::new(crate::nzu!(3));

        // Test buffer filling
        assert_eq!(buffer.process(1.0), 0.0); // Not full yet
        assert_eq!(buffer.process(2.0), 0.0); // Not full yet
        assert_eq!(buffer.process(3.0), 0.0); // Not full yet
        assert_eq!(buffer.process(4.0), 1.0); // Now returns delayed sample
        assert_eq!(buffer.process(5.0), 2.0); // Returns next delayed sample
    }

    #[test]
    fn test_db_linear_conversion() {
        let linear: f64 = 0.5;
        let db = linear_to_db(linear);
        let back_to_linear = db_to_linear(db);

        assert!((linear - back_to_linear).abs() < 1e-10);
    }

    #[test]
    fn test_compression_gain_calculation() {
        let gain_reduction: f64 = calculate_compression_gain(
            -6.0,  // input_level_db
            -12.0, // threshold_db
            4.0,   // ratio
            KneeType::Hard,
            2.0, // knee_width_db
        );

        // With 6dB overshoot and 4:1 ratio, we should get 4.5dB gain reduction
        assert!((gain_reduction - 4.5).abs() < 1e-10);
    }

    #[test]
    fn test_limiting_gain_calculation() {
        let gain_reduction: f64 = calculate_limiting_gain(
            -0.5, // input_level_db
            -1.0, // ceiling_db
            KneeType::Hard,
            1.0, // knee_width_db
        );

        // With 0.5dB overshoot, we should get 0.5dB gain reduction
        assert!((gain_reduction - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_compressor_config_validation() {
        let mut config: CompressorConfig = CompressorConfig::new();

        // Valid config should pass
        assert!(config.validate(44100.0).is_ok());

        // Invalid threshold should fail
        config.threshold_db = 5.0;
        assert!(config.validate(44100.0).is_err());

        // Fix threshold, test invalid ratio
        config.threshold_db = -12.0;
        config.ratio = 0.5;
        assert!(config.validate(44100.0).is_err());
    }

    #[test]
    fn test_limiter_config_validation() {
        let config = LimiterConfig::default();

        let config = config.validate(44100.0);
        // Valid config should pass
        assert!(
            config.is_ok(),
            "Valid limiter config failed validation: {:?}",
            config
        );
        let mut config = config.expect("safe unwrap");

        // Invalid ceiling should fail

        config.ceiling_db = 5.0;
        let config = config.validate(44100.0);

        assert!(
            config.is_err(),
            "Invalid ceiling_db passed validation: {:?}",
            config
        );

        let mut config = LimiterConfig::default();
        // Fix ceiling, test invalid attack time
        config.ceiling_db = -1.0;
        config.attack_ms = 0.0;
        assert!(config.validate(44100.0).is_err());
    }

    #[test]
    fn test_multi_channel_compressor() {
        let data = ndarray::Array2::from_shape_vec(
            (2, 5),
            vec![0.1f32, 0.8, 0.2, 0.9, 0.1, 0.2f32, 0.7, 0.3, 0.8, 0.2],
        )
        .unwrap();
        let mut audio = AudioSamples::new_multi_channel(data.into(), sample_rate!(44100)).unwrap();

        let config = CompressorConfig::new();
        let result = audio.apply_compressor(&config);

        assert!(result.is_ok());
    }

    #[test]
    fn test_multi_channel_limiter() {
        let data = ndarray::Array2::from_shape_vec(
            (2, 5),
            vec![0.1f32, 0.8, 0.2, 0.9, 0.1, 0.2f32, 0.7, 0.3, 0.8, 0.2],
        )
        .unwrap();
        let mut audio = AudioSamples::new_multi_channel(data.into(), sample_rate!(44100)).unwrap();

        let config = LimiterConfig::default();
        let result = audio.apply_limiter(&config);

        assert!(result.is_ok());
    }

    #[test]
    fn test_compressor_presets() {
        let data = Array1::from_vec(vec![0.1f32, 0.8, 0.2, 0.9, 0.1]);
        let mut audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();

        // Test different presets
        let vocal_config = CompressorConfig::vocal();
        assert!(audio.apply_compressor(&vocal_config).is_ok());

        let drum_config = CompressorConfig::drum();
        assert!(audio.apply_compressor(&drum_config).is_ok());

        let bus_config = CompressorConfig::bus();
        assert!(audio.apply_compressor(&bus_config).is_ok());
    }

    #[test]
    fn test_limiter_presets() {
        let data = Array1::from_vec(vec![0.1f32, 0.8, 0.2, 0.9, 0.1]);
        let mut audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();

        // Test different presets
        let transparent_config = LimiterConfig::transparent();
        assert!(audio.apply_limiter(&transparent_config).is_ok());

        let mastering_config = LimiterConfig::mastering();
        assert!(audio.apply_limiter(&mastering_config).is_ok());

        let broadcast_config = LimiterConfig::broadcast();
        assert!(audio.apply_limiter(&broadcast_config).is_ok());
    }
}
