//! Dynamic range processing implementation.
//!
//! This module provides professional-grade compressor and limiter algorithms
//! with support for side-chain processing, lookahead limiting, and various
//! detection methods. The implementations focus on mathematical correctness
//! and professional audio quality.

use crate::operations::traits::AudioDynamicRange;
use crate::operations::types::{CompressorConfig, DynamicRangeMethod, KneeType, LimiterConfig};
use crate::repr::AudioData;
use crate::{
    AudioSample, AudioSampleError, AudioSampleResult, AudioSamples, AudioTypeConversion, ConvertTo,
    I24, RealFloat, to_precision,
};
use std::collections::VecDeque;

/// Envelope follower for attack and release processing.
///
/// This struct implements a first-order RC-style envelope follower
/// with separate attack and release time constants.
#[derive(Debug, Clone)]
pub struct EnvelopeFollower<F: RealFloat> {
    /// Current envelope value
    envelope: F,
    /// Attack coefficient (0.0 to 1.0)
    attack_coeff: F,
    /// Release coefficient (0.0 to 1.0)
    release_coeff: F,
    /// Current detector state
    detector_state: F,
    /// RMS window for RMS detection
    rms_window: VecDeque<F>,
    /// RMS window size
    rms_window_size: usize,
    /// RMS window sum
    rms_sum: F,
}

impl<F: RealFloat> EnvelopeFollower<F> {
    /// Create a new envelope follower.
    ///
    /// # Arguments
    /// * `attack_ms` - Attack time in milliseconds
    /// * `release_ms` - Release time in milliseconds
    /// * `sample_rate` - Sample rate in Hz
    /// * `detection_method` - Detection method (RMS, Peak, or Hybrid)
    pub fn new(
        attack_ms: F,
        release_ms: F,
        sample_rate: F,
        detection_method: DynamicRangeMethod,
    ) -> Self {
        let attack_coeff = if attack_ms > F::zero() {
            (-F::one() / (attack_ms * to_precision::<F, _>(0.001) * sample_rate)).exp()
        } else {
            F::zero()
        };

        let release_coeff = if release_ms > F::zero() {
            (-F::one() / (release_ms * to_precision::<F, _>(0.001) * sample_rate)).exp()
        } else {
            F::zero()
        };

        // RMS window size: 10ms for RMS detection
        let rms_window_size = match detection_method {
            DynamicRangeMethod::Rms | DynamicRangeMethod::Hybrid => (to_precision::<F, _>(0.01)
                * sample_rate)
                .max(F::one())
                .to_usize()
                .expect("Should not fail"),
            DynamicRangeMethod::Peak => 1,
        };

        Self {
            envelope: F::zero(),
            attack_coeff,
            release_coeff,
            detector_state: F::zero(),
            rms_window: VecDeque::with_capacity(rms_window_size),
            rms_window_size,
            rms_sum: F::zero(),
        }
    }

    /// Process a sample through the envelope follower.
    ///
    /// # Arguments
    /// * `input` - Input sample value
    /// * `detection_method` - Detection method to use
    ///
    /// # Returns
    /// Envelope value
    pub fn process(&mut self, input: F, detection_method: DynamicRangeMethod) -> F {
        let detector_value = match detection_method {
            DynamicRangeMethod::Peak => input.abs(),
            DynamicRangeMethod::Rms => {
                // Add new sample to RMS window
                let sample_squared = input * input;
                self.rms_window.push_back(sample_squared);
                self.rms_sum += sample_squared;

                // Remove old samples if window is full
                if self.rms_window.len() > self.rms_window_size {
                    if let Some(old_sample) = self.rms_window.pop_front() {
                        self.rms_sum -= old_sample;
                    }
                }

                // Calculate RMS
                if !self.rms_window.is_empty() {
                    (self.rms_sum / to_precision::<F, _>(self.rms_window.len())).sqrt()
                } else {
                    F::zero()
                }
            }
            DynamicRangeMethod::Hybrid => {
                // Hybrid: max of peak and RMS
                let peak = input.abs();

                // RMS calculation
                let sample_squared = input * input;
                self.rms_window.push_back(sample_squared);
                self.rms_sum += sample_squared;

                if self.rms_window.len() > self.rms_window_size {
                    if let Some(old_sample) = self.rms_window.pop_front() {
                        self.rms_sum -= old_sample;
                    }
                }

                let rms = if !self.rms_window.is_empty() {
                    (self.rms_sum / to_precision::<F, _>(self.rms_window.len())).sqrt()
                } else {
                    F::zero()
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
        self.envelope = coeff * self.envelope + (F::one() - coeff) * detector_value;
        self.envelope
    }

    /// Reset the envelope follower state.
    pub fn reset(&mut self) {
        self.envelope = F::zero();
        self.detector_state = F::zero();
        self.rms_window.clear();
        self.rms_sum = F::zero();
    }
}

/// Lookahead buffer for limiting applications.
///
/// This struct provides a delay buffer that allows the processor to
/// "look ahead" at upcoming samples to prevent peaks before they occur.
#[derive(Debug, Clone)]
pub struct LookaheadBuffer<F: RealFloat> {
    /// Circular buffer for samples
    buffer: Vec<F>,
    /// Write position in buffer
    write_pos: usize,
    /// Read position in buffer
    read_pos: usize,
    /// Buffer size
    size: usize,
    /// Whether buffer is full
    is_full: bool,
}

impl<F: RealFloat> LookaheadBuffer<F> {
    /// Create a new lookahead buffer.
    ///
    /// # Arguments
    /// * `lookahead_samples` - Number of samples to look ahead
    pub fn new(lookahead_samples: usize) -> Self {
        let size = lookahead_samples.max(1);
        Self {
            buffer: vec![F::zero(); size],
            write_pos: 0,
            read_pos: 0,
            size,
            is_full: false,
        }
    }

    /// Add a sample to the buffer and get the delayed sample.
    ///
    /// # Arguments
    /// * `input` - Input sample
    ///
    /// # Returns
    /// Delayed sample (or 0.0 if buffer not full yet)
    pub fn process(&mut self, input: F) -> F {
        // For a lookahead buffer, we need to return the oldest sample
        // when the buffer is full, otherwise return 0.0

        let output = if self.is_full {
            // Return the oldest sample (at read position)
            self.buffer[self.read_pos]
        } else {
            F::zero()
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

    /// Get the current sample at the lookahead position.
    ///
    /// # Returns
    /// Current sample value
    pub fn peek(&self) -> F {
        self.buffer[self.write_pos]
    }

    /// Reset the buffer.
    pub fn reset(&mut self) {
        self.buffer.fill(F::zero());
        self.write_pos = 0;
        self.read_pos = 0;
        self.is_full = false;
    }
}

/// Convert time in milliseconds to sample count.
fn ms_to_samples<F: RealFloat>(ms: F, sample_rate: F) -> usize {
    (ms * to_precision::<F, _>(0.001) * sample_rate)
        .round()
        .to_usize()
        .expect("Should not fail")
}

/// Convert linear amplitude to decibels.
fn linear_to_db<F: RealFloat>(linear: F) -> F {
    if linear > F::zero() {
        to_precision::<F, _>(20.0) * linear.log10()
    } else {
        to_precision::<F, _>(-80.0) // Floor at -80 dB
    }
}

/// Convert decibels to linear amplitude.
fn db_to_linear<F: RealFloat>(db: F) -> F {
    to_precision::<F, _>(10.0).powf(db / to_precision::<F, _>(20.0))
}

/// Calculate compression gain reduction.
///
/// # Arguments
/// * `input_level_db` - Input level in dB
/// * `threshold_db` - Threshold in dB
/// * `ratio` - Compression ratio
/// * `knee_type` - Knee type (Hard or Soft)
/// * `knee_width_db` - Knee width in dB
///
/// # Returns
/// Gain reduction in dB (positive values indicate reduction)
fn calculate_compression_gain<F: RealFloat>(
    input_level_db: F,
    threshold_db: F,
    ratio: F,
    knee_type: KneeType,
    knee_width_db: F,
) -> F {
    let overshoot = input_level_db - threshold_db;

    if overshoot <= F::zero() {
        return F::zero();
    }

    match knee_type {
        KneeType::Hard => {
            // Hard knee: abrupt transition
            overshoot - overshoot / ratio
        }
        KneeType::Soft => {
            // Soft knee: smooth transition
            let half_knee = knee_width_db / to_precision::<F, _>(2.0);

            if overshoot <= half_knee {
                // In the knee region
                let knee_ratio = overshoot / half_knee;
                let smooth_ratio = F::one() + (ratio - F::one()) * knee_ratio * knee_ratio;
                overshoot - overshoot / smooth_ratio
            } else {
                // Above the knee
                let knee_gain = half_knee - half_knee / ratio;
                knee_gain + (overshoot - half_knee) - (overshoot - half_knee) / ratio
            }
        }
    }
}

/// Calculate limiting gain reduction.
///
/// # Arguments
/// * `input_level_db` - Input level in dB
/// * `ceiling_db` - Ceiling in dB
/// * `knee_type` - Knee type (Hard or Soft)
/// * `knee_width_db` - Knee width in dB
///
/// # Returns
/// Gain reduction in dB (positive values indicate reduction)
fn calculate_limiting_gain<F: RealFloat>(
    input_level_db: F,
    ceiling_db: F,
    knee_type: KneeType,
    knee_width_db: F,
) -> F {
    let overshoot = input_level_db - ceiling_db;

    if overshoot <= F::zero() {
        return F::zero();
    }

    match knee_type {
        KneeType::Hard => {
            // Hard limiting: complete attenuation of overshoot
            overshoot
        }
        KneeType::Soft => {
            // Soft limiting: smooth transition to full limiting
            let half_knee = knee_width_db / to_precision::<F, _>(2.0);

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

impl<'a, T: AudioSample> AudioDynamicRange<T> for AudioSamples<'a, T>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    AudioSamples<'a, T>: AudioTypeConversion<'a, T>,
{
    fn apply_compressor<F>(
        &mut self,
        config: &CompressorConfig<F>,
        sample_rate: F,
    ) -> AudioSampleResult<()>
    where
        F: RealFloat + ConvertTo<T>,
        T: ConvertTo<F>,
    {
        // Validate configuration
        config.validate(sample_rate)?;

        // Calculate lookahead buffer size
        let lookahead_samples = ms_to_samples(config.lookahead_ms, sample_rate);

        match &mut self.data {
            AudioData::Mono(samples) => {
                let mut envelope_follower = EnvelopeFollower::new(
                    config.attack_ms,
                    config.release_ms,
                    sample_rate,
                    config.detection_method,
                );

                let mut lookahead_buffer = LookaheadBuffer::new(lookahead_samples);
                let mut gain_reductions = Vec::with_capacity(samples.len());

                // Process each sample
                for &sample in samples.iter() {
                    let sample_f: F = sample.convert_to()?;

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
                for i in 0..samples.len() {
                    let sample_f: F = samples[i].convert_to()?;
                    let delayed_sample = lookahead_buffer.process(sample_f);

                    if i >= lookahead_samples {
                        let gain_reduction_db = gain_reductions[i - lookahead_samples];
                        let gain_linear = db_to_linear(-gain_reduction_db);
                        let makeup_gain = db_to_linear(config.makeup_gain_db);

                        let output_sample = delayed_sample * gain_linear * makeup_gain;
                        samples[i - lookahead_samples] = output_sample.convert_to()?;
                    }
                }

                // Process remaining samples in lookahead buffer
                for i in 0..lookahead_samples {
                    let delayed_sample = lookahead_buffer.process(F::zero());

                    // Check for overflow before subtraction
                    if lookahead_samples <= samples.len() {
                        let sample_idx = samples.len() - lookahead_samples + i;

                        if sample_idx < samples.len() {
                            let gain_reduction_db = gain_reductions[sample_idx];
                            let gain_linear = db_to_linear(-gain_reduction_db);
                            let makeup_gain = db_to_linear(config.makeup_gain_db);

                            let output_sample = delayed_sample * gain_linear * makeup_gain;
                            samples[sample_idx] = output_sample.convert_to()?;
                        }
                    }
                }
            }
            AudioData::Multi(samples) => {
                let num_channels = samples.nrows();
                let num_samples = samples.ncols();

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
                        let sample_f: F = samples[[channel, sample_idx]].convert_to()?;

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
                        let sample_f: F = samples[[channel, sample_idx]].convert_to()?;
                        let delayed_sample = lookahead_buffer.process(sample_f);

                        if sample_idx >= lookahead_samples {
                            let gain_reduction_db = gain_reductions[sample_idx - lookahead_samples];
                            let gain_linear = db_to_linear(-gain_reduction_db);
                            let makeup_gain = db_to_linear(config.makeup_gain_db);

                            let output_sample = delayed_sample * gain_linear * makeup_gain;
                            samples[[channel, sample_idx - lookahead_samples]] =
                                output_sample.convert_to()?;
                        }
                    }

                    // Process remaining samples in lookahead buffer
                    for i in 0..lookahead_samples {
                        let delayed_sample = lookahead_buffer.process(F::zero());

                        // Check for overflow before subtraction
                        if lookahead_samples <= num_samples {
                            let sample_idx = num_samples - lookahead_samples + i;

                            if sample_idx < num_samples {
                                let gain_reduction_db = gain_reductions[sample_idx];
                                let gain_linear = db_to_linear(-gain_reduction_db);
                                let makeup_gain = db_to_linear(config.makeup_gain_db);

                                let output_sample = delayed_sample * gain_linear * makeup_gain;
                                samples[[channel, sample_idx]] = output_sample.convert_to()?;
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    fn apply_limiter<F>(
        &mut self,
        config: &LimiterConfig<F>,
        sample_rate: F,
    ) -> AudioSampleResult<()>
    where
        F: RealFloat + ConvertTo<T>,
        T: ConvertTo<F>,
    {
        // Validate configuration
        config.validate(sample_rate)?;

        // Calculate lookahead buffer size
        let lookahead_samples = ms_to_samples(config.lookahead_ms, sample_rate);

        match &mut self.data {
            AudioData::Mono(samples) => {
                let mut envelope_follower = EnvelopeFollower::new(
                    config.attack_ms,
                    config.release_ms,
                    sample_rate,
                    config.detection_method,
                );

                let mut lookahead_buffer = LookaheadBuffer::new(lookahead_samples);
                let mut gain_reductions = Vec::with_capacity(samples.len());

                // Process each sample
                for &sample in samples.iter() {
                    let sample_f: F = sample.convert_to()?;

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
                for i in 0..samples.len() {
                    let sample_f: F = samples[i].convert_to()?;
                    let delayed_sample = lookahead_buffer.process(sample_f);

                    if i >= lookahead_samples {
                        let gain_reduction_db = gain_reductions[i - lookahead_samples];
                        let gain_linear = db_to_linear(-gain_reduction_db);

                        let output_sample = delayed_sample * gain_linear;
                        samples[i - lookahead_samples] = output_sample.convert_to()?;
                    }
                }

                // Process remaining samples in lookahead buffer
                for i in 0..lookahead_samples {
                    let delayed_sample = lookahead_buffer.process(F::zero());

                    // Check for overflow before subtraction
                    if lookahead_samples <= samples.len() {
                        let sample_idx = samples.len() - lookahead_samples + i;

                        if sample_idx < samples.len() {
                            let gain_reduction_db = gain_reductions[sample_idx];
                            let gain_linear = db_to_linear(-gain_reduction_db);

                            let output_sample = delayed_sample * gain_linear;
                            samples[sample_idx] = output_sample.convert_to()?;
                        }
                    }
                }
            }
            AudioData::Multi(samples) => {
                let num_channels = samples.nrows();
                let num_samples = samples.ncols();

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
                        let sample_f: F = samples[[channel, sample_idx]].convert_to()?;

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
                        let sample_f: F = samples[[channel, sample_idx]].convert_to()?;
                        let delayed_sample = lookahead_buffer.process(sample_f);

                        if sample_idx >= lookahead_samples {
                            let gain_reduction_db = gain_reductions[sample_idx - lookahead_samples];
                            let gain_linear = db_to_linear(-gain_reduction_db);

                            let output_sample = delayed_sample * gain_linear;
                            samples[[channel, sample_idx - lookahead_samples]] =
                                output_sample.convert_to()?;
                        }
                    }

                    // Process remaining samples in lookahead buffer
                    for i in 0..lookahead_samples {
                        let delayed_sample = lookahead_buffer.process(F::zero());

                        // Check for overflow before subtraction
                        if lookahead_samples <= num_samples {
                            let sample_idx = num_samples - lookahead_samples + i;

                            if sample_idx < num_samples {
                                let gain_reduction_db = gain_reductions[sample_idx];
                                let gain_linear = db_to_linear(-gain_reduction_db);

                                let output_sample = delayed_sample * gain_linear;
                                samples[[channel, sample_idx]] = output_sample.convert_to()?;
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    fn apply_compressor_sidechain<F>(
        &mut self,
        config: &CompressorConfig<F>,
        sidechain_signal: &Self,
        sample_rate: F,
    ) -> AudioSampleResult<()>
    where
        F: RealFloat + ConvertTo<T>,
        T: ConvertTo<F>,
    {
        // Validate configuration
        config.validate(sample_rate)?;

        if !config.side_chain.enabled {
            return Err(AudioSampleError::InvalidParameter(
                "Side-chain is not enabled in configuration".to_string(),
            ));
        }

        // For now, implement basic side-chain by using the sidechain signal for detection
        // but applying gain reduction to the main signal
        // This is a simplified implementation - a full implementation would include
        // filtering of the sidechain signal according to the configuration

        match (&mut self.data, &sidechain_signal.data) {
            (AudioData::Mono(main_samples), AudioData::Mono(sc_samples)) => {
                if main_samples.len() != sc_samples.len() {
                    return Err(AudioSampleError::InvalidParameter(
                        "Main signal and sidechain signal must have the same length".to_string(),
                    ));
                }

                let mut envelope_follower = EnvelopeFollower::new(
                    config.attack_ms,
                    config.release_ms,
                    sample_rate,
                    config.detection_method,
                );

                let lookahead_samples = ms_to_samples(config.lookahead_ms, sample_rate);
                let mut lookahead_buffer = LookaheadBuffer::new(lookahead_samples);
                let mut gain_reductions = Vec::with_capacity(main_samples.len());

                // Process sidechain signal to get gain reductions
                for &sc_sample in sc_samples.iter() {
                    let sc_f: F = sc_sample.convert_to()?;

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
                for i in 0..main_samples.len() {
                    let sample_f: F = main_samples[i].convert_to()?;
                    let delayed_sample = lookahead_buffer.process(sample_f);

                    if i >= lookahead_samples {
                        let gain_reduction_db = gain_reductions[i - lookahead_samples];
                        let gain_linear = db_to_linear(-gain_reduction_db);
                        let makeup_gain = db_to_linear(config.makeup_gain_db);

                        let output_sample = delayed_sample * gain_linear * makeup_gain;
                        main_samples[i - lookahead_samples] = output_sample.convert_to()?;
                    }
                }

                // Process remaining samples in lookahead buffer
                for i in 0..lookahead_samples {
                    let delayed_sample = lookahead_buffer.process(F::zero());
                    let sample_idx = main_samples.len() - lookahead_samples + i;

                    if sample_idx < main_samples.len() {
                        let gain_reduction_db = gain_reductions[sample_idx];
                        let gain_linear = db_to_linear(-gain_reduction_db);
                        let makeup_gain = db_to_linear(config.makeup_gain_db);

                        let output_sample = delayed_sample * gain_linear * makeup_gain;
                        main_samples[sample_idx] = output_sample.convert_to()?;
                    }
                }
            }
            _ => {
                return Err(AudioSampleError::InvalidParameter(
                    "Side-chain processing with multi-channel audio not yet implemented"
                        .to_string(),
                ));
            }
        }

        Ok(())
    }

    fn apply_limiter_sidechain<F>(
        &mut self,
        config: &LimiterConfig<F>,
        sidechain_signal: &Self,
        sample_rate: F,
    ) -> AudioSampleResult<()>
    where
        F: RealFloat + ConvertTo<T>,
        T: ConvertTo<F>,
    {
        // Validate configuration
        config.validate(sample_rate)?;
        if !config.side_chain.enabled {
            return Err(AudioSampleError::InvalidParameter(
                "Side-chain is not enabled in configuration".to_string(),
            ));
        }

        // Similar to compressor sidechain but with limiting gain calculation
        match (&mut self.data, &sidechain_signal.data) {
            (AudioData::Mono(main_samples), AudioData::Mono(sc_samples)) => {
                if main_samples.len() != sc_samples.len() {
                    return Err(AudioSampleError::InvalidParameter(
                        "Main signal and sidechain signal must have the same length".to_string(),
                    ));
                }

                let mut envelope_follower = EnvelopeFollower::new(
                    config.attack_ms,
                    config.release_ms,
                    sample_rate,
                    config.detection_method,
                );

                let lookahead_samples = ms_to_samples(config.lookahead_ms, sample_rate);
                let mut lookahead_buffer = LookaheadBuffer::new(lookahead_samples);
                let mut gain_reductions = Vec::with_capacity(main_samples.len());

                // Process sidechain signal to get gain reductions
                for &sc_sample in sc_samples.iter() {
                    let sc_f: F = sc_sample.convert_to()?;

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
                for i in 0..main_samples.len() {
                    let sample_f: F = main_samples[i].convert_to()?;
                    let delayed_sample = lookahead_buffer.process(sample_f);

                    if i >= lookahead_samples {
                        let gain_reduction_db = gain_reductions[i - lookahead_samples];
                        let gain_linear = db_to_linear(-gain_reduction_db);

                        let output_sample = delayed_sample * gain_linear;
                        main_samples[i - lookahead_samples] = output_sample.convert_to()?;
                    }
                }

                // Process remaining samples in lookahead buffer
                for i in 0..lookahead_samples {
                    let delayed_sample = lookahead_buffer.process(F::zero());
                    let sample_idx = main_samples.len() - lookahead_samples + i;

                    if sample_idx < main_samples.len() {
                        let gain_reduction_db = gain_reductions[sample_idx];
                        let gain_linear = db_to_linear(-gain_reduction_db);

                        let output_sample = delayed_sample * gain_linear;
                        main_samples[sample_idx] = output_sample.convert_to()?;
                    }
                }
            }
            _ => {
                return Err(AudioSampleError::InvalidParameter(
                    "Side-chain processing with multi-channel audio not yet implemented"
                        .to_string(),
                ));
            }
        }

        Ok(())
    }

    fn get_compression_curve<F: RealFloat>(
        &self,
        config: &CompressorConfig<F>,
        input_levels_db: &[F],
        _sample_rate: F,
    ) -> AudioSampleResult<Vec<F>> {
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

    fn get_gain_reduction<F>(
        &self,
        config: &CompressorConfig<F>,
        sample_rate: F,
    ) -> AudioSampleResult<Vec<F>>
    where
        F: RealFloat + ConvertTo<T>,
        T: ConvertTo<F>,
    {
        match &self.data {
            AudioData::Mono(samples) => {
                let mut envelope_follower = EnvelopeFollower::new(
                    config.attack_ms,
                    config.release_ms,
                    sample_rate,
                    config.detection_method,
                );

                let mut gain_reductions = Vec::with_capacity(samples.len());

                for &sample in samples.iter() {
                    let sample_f: F = sample.convert_to()?;
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
                let num_samples = samples.ncols();
                let mut envelope_follower = EnvelopeFollower::new(
                    config.attack_ms,
                    config.release_ms,
                    sample_rate,
                    config.detection_method,
                );

                let mut gain_reductions = Vec::with_capacity(num_samples);

                for sample_idx in 0..num_samples {
                    let sample_f: F = samples[[0, sample_idx]].convert_to()?;
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

    fn apply_gate<F>(
        &mut self,
        threshold_db: F,
        ratio: F,
        attack_ms: F,
        release_ms: F,
        sample_rate: F,
    ) -> AudioSampleResult<()>
    where
        F: RealFloat + ConvertTo<T>,
        T: ConvertTo<F>,
    {
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
                    let sample_f: F = sample.convert_to()?;
                    let envelope = envelope_follower.process(sample_f, DynamicRangeMethod::Peak);
                    let envelope_db = linear_to_db(envelope);

                    // Gate logic: attenuate if below threshold
                    let gain_reduction_db = if envelope_db < threshold_db {
                        let undershoot = threshold_db - envelope_db;
                        undershoot * (ratio - F::one()) / ratio
                    } else {
                        F::zero()
                    };

                    let gain_linear = db_to_linear(-gain_reduction_db);
                    let output_sample = sample_f * gain_linear;

                    *sample = output_sample.convert_to()?;
                }
            }
            AudioData::Multi(samples) => {
                let num_channels = samples.nrows();
                let num_samples = samples.ncols();

                for channel in 0..num_channels {
                    let mut envelope_follower = EnvelopeFollower::new(
                        attack_ms,
                        release_ms,
                        sample_rate,
                        DynamicRangeMethod::Peak,
                    );

                    for sample_idx in 0..num_samples {
                        let sample_f: F = samples[[channel, sample_idx]].convert_to()?;
                        let envelope =
                            envelope_follower.process(sample_f, DynamicRangeMethod::Peak);
                        let envelope_db = linear_to_db(envelope);

                        // Gate logic: attenuate if below threshold
                        let gain_reduction_db = if envelope_db < threshold_db {
                            let undershoot = threshold_db - envelope_db;
                            undershoot * (ratio - F::one()) / ratio
                        } else {
                            F::zero()
                        };

                        let gain_linear = db_to_linear(-gain_reduction_db);
                        let output_sample = sample_f * gain_linear;

                        samples[[channel, sample_idx]] = output_sample.convert_to()?;
                    }
                }
            }
        }

        Ok(())
    }

    fn apply_expander<F>(
        &mut self,
        threshold_db: F,
        ratio: F,
        attack_ms: F,
        release_ms: F,
        sample_rate: F,
    ) -> AudioSampleResult<()>
    where
        F: RealFloat + ConvertTo<T>,
        T: ConvertTo<F>,
    {
        // Expander increases dynamic range by expanding signals below threshold

        match &mut self.data {
            AudioData::Mono(samples) => {
                let mut envelope_follower = EnvelopeFollower::<F>::new(
                    attack_ms,
                    release_ms,
                    sample_rate,
                    DynamicRangeMethod::Rms, // Use RMS detection for expanders
                );

                for sample in samples.iter_mut() {
                    let sample_f: F = sample.convert_to()?;
                    let envelope = envelope_follower.process(sample_f, DynamicRangeMethod::Rms);
                    let envelope_db = linear_to_db(envelope);

                    // Expander logic: amplify the difference below threshold
                    let gain_change_db = if envelope_db < threshold_db {
                        let undershoot = threshold_db - envelope_db;
                        undershoot * (ratio - F::one()) // Expand by ratio
                    } else {
                        F::one()
                    };

                    let gain_linear = db_to_linear(-gain_change_db); // Negative because we're reducing level
                    let output_sample = sample_f * gain_linear;

                    *sample = output_sample.convert_to()?;
                }
            }
            AudioData::Multi(samples) => {
                let num_channels = samples.nrows();
                let num_samples = samples.ncols();

                for channel in 0..num_channels {
                    let mut envelope_follower = EnvelopeFollower::new(
                        attack_ms,
                        release_ms,
                        sample_rate,
                        DynamicRangeMethod::Rms,
                    );

                    for sample_idx in 0..num_samples {
                        let sample_f: F = samples[[channel, sample_idx]].convert_to()?;
                        let envelope = envelope_follower.process(sample_f, DynamicRangeMethod::Rms);
                        let envelope_db = linear_to_db(envelope);

                        // Expander logic: amplify the difference below threshold
                        let gain_change_db = if envelope_db < threshold_db {
                            let undershoot = threshold_db - envelope_db;
                            undershoot * (ratio - F::one()) // Expand by ratio
                        } else {
                            F::zero()
                        };

                        let gain_linear = db_to_linear(-gain_change_db); // Negative because we're reducing level
                        let output_sample = sample_f * gain_linear;

                        samples[[channel, sample_idx]] = output_sample.convert_to()?;
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
    use ndarray::Array1;

    #[test]
    fn test_compressor_basic() {
        let data = Array1::from_vec(vec![0.1f32, 0.8, 0.2, 0.9, 0.1]);
        let mut audio = AudioSamples::new_mono(data, 44100);

        let config = CompressorConfig::new();
        let result = audio.apply_compressor(&config, 44100.0);

        assert!(result.is_ok());
    }

    #[test]
    fn test_limiter_basic() {
        let data = Array1::from_vec(vec![0.1f32, 0.8, 0.2, 0.9, 0.1]);
        let mut audio = AudioSamples::new_mono(data, 44100);

        let config = LimiterConfig::new();
        let result = audio.apply_limiter(&config, 44100.0);

        assert!(result.is_ok());
    }

    #[test]
    fn test_compression_curve() {
        let data = Array1::from_vec(vec![0.1f32, 0.2, 0.3, 0.4, 0.5]);
        let audio = AudioSamples::new_mono(data, 44100);

        let config = CompressorConfig::new();
        let input_levels = vec![-40.0, -30.0, -20.0, -10.0, 0.0];
        let result = audio.get_compression_curve(&config, &input_levels, 44100.0);

        assert!(result.is_ok());
        let output_levels = result.unwrap();
        assert_eq!(output_levels.len(), input_levels.len());

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
        let audio = AudioSamples::new_mono(data, 44100);

        let config = CompressorConfig::new();
        let result = audio.get_gain_reduction(&config, 44100.0);

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
        let mut audio = AudioSamples::new_mono(data, 44100);

        let result = audio.apply_gate(-20.0, 10.0, 1.0, 10.0, 44100.0);

        assert!(result.is_ok());
    }

    #[test]
    fn test_expander_basic() {
        let data = Array1::from_vec(vec![0.1f32, 0.8, 0.2, 0.9, 0.1]);
        let mut audio = AudioSamples::new_mono(data, 44100);

        let result = audio.apply_expander(-20.0, 2.0, 1.0, 10.0, 44100.0);

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
        let mut buffer = LookaheadBuffer::new(3);

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
        let mut config: CompressorConfig<f64> = CompressorConfig::new();

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
        let mut config = LimiterConfig::new();

        // Valid config should pass
        assert!(config.validate(44100.0).is_ok());

        // Invalid ceiling should fail
        config.ceiling_db = 5.0;
        assert!(config.validate(44100.0).is_err());

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
        let mut audio = AudioSamples::new_multi_channel(data.into(), 44100);

        let config = CompressorConfig::new();
        let result = audio.apply_compressor(&config, 44100.0);

        assert!(result.is_ok());
    }

    #[test]
    fn test_multi_channel_limiter() {
        let data = ndarray::Array2::from_shape_vec(
            (2, 5),
            vec![0.1f32, 0.8, 0.2, 0.9, 0.1, 0.2f32, 0.7, 0.3, 0.8, 0.2],
        )
        .unwrap();
        let mut audio = AudioSamples::new_multi_channel(data.into(), 44100);

        let config = LimiterConfig::new();
        let result = audio.apply_limiter(&config, 44100.0);

        assert!(result.is_ok());
    }

    #[test]
    fn test_compressor_presets() {
        let data = Array1::from_vec(vec![0.1f32, 0.8, 0.2, 0.9, 0.1]);
        let mut audio = AudioSamples::new_mono(data, 44100);

        // Test different presets
        let vocal_config = CompressorConfig::vocal();
        assert!(audio.apply_compressor(&vocal_config, 44100.0).is_ok());

        let drum_config = CompressorConfig::drum();
        assert!(audio.apply_compressor(&drum_config, 44100.0).is_ok());

        let bus_config = CompressorConfig::bus();
        assert!(audio.apply_compressor(&bus_config, 44100.0).is_ok());
    }

    #[test]
    fn test_limiter_presets() {
        let data = Array1::from_vec(vec![0.1f32, 0.8, 0.2, 0.9, 0.1]);
        let mut audio = AudioSamples::new_mono(data, 44100);

        // Test different presets
        let transparent_config = LimiterConfig::transparent();
        assert!(audio.apply_limiter(&transparent_config, 44100.0).is_ok());

        let mastering_config = LimiterConfig::mastering();
        assert!(audio.apply_limiter(&mastering_config, 44100.0).is_ok());

        let broadcast_config = LimiterConfig::broadcast();
        assert!(audio.apply_limiter(&broadcast_config, 44100.0).is_ok());
    }
}
