//! AudioEditing trait implementation for Python bindings.
//!
//! This module provides both in-place and functional audio editing operations
//! following pandas-style conventions. Some operations are inherently functional
//! (like reverse, trim) while others support both modes (like fade operations).

use super::{PyAudioSamples, utils::*};
use crate::operations::{AudioEditing, AudioProcessing, types::{FadeCurve, PerturbationConfig, PerturbationMethod, NoiseColor}};
use pyo3::prelude::*;

impl PyAudioSamples {
    /// Parse fade curve from string
    pub(crate) fn parse_fade_curve_impl(curve: &str) -> PyResult<FadeCurve> {
        match curve.to_lowercase().as_str() {
            "linear" => Ok(FadeCurve::Linear),
            "exponential" | "exp" => Ok(FadeCurve::Exponential),
            "logarithmic" | "log" => Ok(FadeCurve::Logarithmic),
            "smoothstep" | "smooth" => Ok(FadeCurve::SmoothStep),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Invalid fade curve: '{}'. Valid options: 'linear', 'exponential', 'logarithmic', 'smoothstep'",
                curve
            ))),
        }
    }

    /// Parse noise color from string
    pub(crate) fn parse_noise_color_impl(noise_color: &str) -> PyResult<NoiseColor> {
        match noise_color.to_lowercase().as_str() {
            "white" => Ok(NoiseColor::White),
            "pink" => Ok(NoiseColor::Pink),
            "brown" | "red" => Ok(NoiseColor::Brown),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Invalid noise color: '{}'. Valid options: 'white', 'pink', 'brown'",
                noise_color
            ))),
        }
    }

    /// Parse perturbation method from string and arguments
    pub(crate) fn parse_perturbation_method_impl(
        method: &str,
        target_snr_db: Option<f64>,
        noise_color: Option<&str>,
        min_gain_db: Option<f64>,
        max_gain_db: Option<f64>,
        cutoff_hz: Option<f64>,
        slope_db_per_octave: Option<f64>,
        semitones: Option<f64>,
        preserve_formants: Option<bool>,
    ) -> PyResult<PerturbationMethod> {
        match method.to_lowercase().as_str() {
            "gaussian_noise" | "noise" => {
                let snr = target_snr_db.unwrap_or(20.0);
                let color_str = noise_color.unwrap_or("white");
                let color = Self::parse_noise_color_impl(color_str)?;
                Ok(PerturbationMethod::gaussian_noise(snr, color))
            },
            "random_gain" | "gain" => {
                let min = min_gain_db.unwrap_or(-3.0);
                let max = max_gain_db.unwrap_or(3.0);
                Ok(PerturbationMethod::random_gain(min, max))
            },
            "high_pass_filter" | "highpass" | "hpf" => {
                let cutoff = cutoff_hz.unwrap_or(80.0);
                if let Some(slope) = slope_db_per_octave {
                    Ok(PerturbationMethod::high_pass_filter_with_slope(cutoff, slope))
                } else {
                    Ok(PerturbationMethod::high_pass_filter(cutoff))
                }
            },
            "pitch_shift" | "pitch" => {
                let shift = semitones.unwrap_or(0.0);
                let formants = preserve_formants.unwrap_or(false);
                Ok(PerturbationMethod::pitch_shift(shift, formants))
            },
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Invalid perturbation method: '{}'. Valid options: 'gaussian_noise', 'random_gain', 'high_pass_filter', 'pitch_shift'",
                method
            ))),
        }
    }
}

impl PyAudioSamples {
    // ==========================================
    // FUNCTIONAL OPERATIONS (structural changes)
    // ==========================================

    /// Reverse the order of audio samples.
    ///
    /// This operation is always functional as it creates a new structure.
    ///
    /// # Returns
    /// New AudioSamples object with reversed sample order
    ///
    /// # Examples
    /// ```python
    /// import numpy as np
    /// import audio_samples as aus
    ///
    /// # Create test signal
    /// ramp = np.linspace(0, 1, 1000)
    /// audio = aus.from_numpy(ramp, sample_rate=44100)
    ///
    /// # Reverse the signal
    /// reversed_audio = audio.reverse()
    /// # Should go from 1 to 0 instead of 0 to 1
    /// ```
    pub(crate) fn reverse_impl(&self) -> PyResult<PyAudioSamples> {
        let reversed = self
            .map_inner(|inner| Ok(inner.reverse()))
            .map_err(map_error)?;
        Ok(reversed)
    }

    /// Reverse the order of audio samples in-place.
    ///
    /// This operation modifies the original object directly.
    /// /// # Examples
    /// ```python
    /// import numpy as np
    /// import audio_samples as aus
    /// /// # Create test signal
    /// ramp = np.linspace(0, 1, 1000)
    /// audio = aus.from_numpy(ramp, sample_rate=44100)
    /// /// # Reverse the signal in-place
    /// audio.reverse_()
    /// # Should go from 1 to 0 instead of 0 to 1
    /// ```
    pub(crate) fn reverse_inplace_impl(&mut self) -> PyResult<()> {
        self.mutate_inner(|inner| inner.reverse_in_place())
            .map_err(map_error)?;
        Ok(())
    }

    /// Extract a segment of audio between start and end times.
    ///
    /// # Arguments
    /// * `start_sec` - Start time in seconds
    /// * `end_sec` - End time in seconds
    /// * `copy` - Whether to copy data (default: True for safety)
    ///
    /// # Returns
    /// New AudioSamples object containing the trimmed segment
    ///
    /// # Examples
    /// ```python
    /// # Extract middle 2 seconds from a 5-second audio clip
    /// trimmed = audio.trim(start_sec=1.5, end_sec=3.5)
    ///
    /// # Extract first second
    /// intro = audio.trim(start_sec=0.0, end_sec=1.0)
    /// ```
    pub(crate) fn trim_impl(
        &self,
        start_sec: f64,
        end_sec: f64,
        _copy: bool,
    ) -> PyResult<PyAudioSamples> {
        let trimmed = self
            .map_inner(|inner| inner.trim(start_sec, end_sec))
            .map_err(map_error)?;
        Ok(trimmed)
    }

    /// Add padding/silence to the beginning and/or end of the audio.
    ///
    /// # Arguments
    /// * `start_sec` - Seconds of padding to add at the beginning
    /// * `end_sec` - Seconds of padding to add at the end
    /// * `value` - Value to use for padding (default: 0.0 for silence)
    ///
    /// # Returns
    /// New AudioSamples object with padding added
    ///
    /// # Examples
    /// ```python
    /// # Add 0.5 seconds of silence at start and end
    /// padded = audio.pad(start_sec=0.5, end_sec=0.5)
    ///
    /// # Add only at the beginning
    /// padded_start = audio.pad(start_sec=1.0, end_sec=0.0)
    ///
    /// # Pad with specific value instead of silence
    /// padded_custom = audio.pad(start_sec=0.1, end_sec=0.1, value=0.1)
    /// ```
    pub(crate) fn pad_impl(
        &self,
        start_sec: f64,
        end_sec: f64,
        value: f64,
    ) -> PyResult<PyAudioSamples> {
        let padded = self
            .map_inner(|inner| inner.pad(start_sec, end_sec, value))
            .map_err(map_error)?;
        Ok(padded)
    }

    /// Split audio into segments of specified duration.
    ///
    /// # Arguments
    /// * `segment_duration` - Duration of each segment in seconds
    /// * `overlap` - Overlap between segments in seconds (default: 0.0)
    ///
    /// # Returns
    /// List of AudioSamples objects, each containing one segment
    ///
    /// # Examples
    /// ```python
    /// # Split into 2-second segments
    /// segments = audio.split(segment_duration=2.0)
    ///
    /// # Split with 50% overlap (1 second segments with 0.5 second overlap)
    /// overlapped = audio.split(segment_duration=1.0, overlap=0.5)
    /// ```
    pub(crate) fn split_impl(
        &self,
        segment_duration: f64,
        _overlap: f64,
    ) -> PyResult<Vec<PyAudioSamples>> {
        let segments = self
            .with_inner(|inner| inner.split(segment_duration))
            .map_err(map_error)?;
        let py_segments: Vec<PyAudioSamples> = segments
            .into_iter()
            .map(PyAudioSamples::from_inner)
            .collect();
        Ok(py_segments)
    }

    /// Repeat the audio signal a specified number of times.
    ///
    /// # Arguments
    /// * `count` - Number of repetitions (2 = play twice, 3 = play three times, etc.)
    ///
    /// # Returns
    /// New AudioSamples object with repeated audio
    ///
    /// # Examples
    /// ```python
    /// # Repeat a short sound 3 times
    /// repeated = audio.repeat(count=3)
    ///
    /// # Create a loop
    /// loop = short_clip.repeat(count=10)
    /// ```
    pub(crate) fn repeat_impl(&self, count: usize) -> PyResult<PyAudioSamples> {
        let repeated = self
            .map_inner(|inner| inner.repeat(count))
            .map_err(map_error)?;
        Ok(repeated)
    }

    /// Remove silence from the beginning and end of the audio.
    ///
    /// # Arguments
    /// * `threshold` - Amplitude threshold below which samples are considered silence
    /// * `min_duration` - Minimum duration of silence to remove (in seconds)
    ///
    /// # Returns
    /// New AudioSamples object with silence trimmed
    ///
    /// # Examples
    /// ```python
    /// # Remove silence with automatic threshold
    /// trimmed = audio.trim_silence(threshold=0.01)
    ///
    /// # More aggressive silence removal
    /// aggressive = audio.trim_silence(threshold=0.001, min_duration=0.1)
    /// ```
    pub(crate) fn trim_silence_impl(
        &self,
        threshold: f64,
        _min_duration: f64,
    ) -> PyResult<PyAudioSamples> {
        let trimmed = self
            .map_inner(|inner| inner.trim_silence(threshold))
            .map_err(map_error)?;
        Ok(trimmed)
    }

    // ==========================================
    // DUAL-MODE OPERATIONS (fade operations)
    // ==========================================

    /// Apply fade-in envelope in-place.
    ///
    /// # Arguments
    /// * `duration` - Fade duration in seconds
    /// * `curve` - Fade curve type ('linear', 'exponential', 'logarithmic', 'smoothstep')
    ///
    /// # Examples
    /// ```python
    /// # Linear fade-in over 0.5 seconds
    /// audio.fade_in_(duration=0.5, curve='linear')
    ///
    /// # Smooth fade-in
    /// audio.fade_in_(duration=1.0, curve='smoothstep')
    /// ```
    pub(crate) fn fade_in_inplace_impl(&mut self, duration: f64, curve: &str) -> PyResult<()> {
        let fade_curve = Self::parse_fade_curve_impl(curve)?;
        self.mutate_inner(|inner| inner.fade_in(duration, fade_curve))
            .map_err(map_error)
    }

    /// Apply fade-in envelope (functional version).
    ///
    /// # Arguments
    /// * `duration` - Fade duration in seconds
    /// * `curve` - Fade curve type
    ///
    /// # Returns
    /// New AudioSamples object with fade-in applied
    ///
    /// # Examples
    /// ```python
    /// # Create faded copy (original unchanged)
    /// faded = audio.fade_in(duration=0.5, curve='exponential')
    /// ```
    pub(crate) fn fade_in_impl(&self, duration: f64, curve: &str) -> PyResult<PyAudioSamples> {
        let mut result = self.copy();
        result.fade_in_inplace_impl(duration, curve)?;
        Ok(result)
    }

    /// Apply fade-out envelope in-place.
    ///
    /// # Arguments
    /// * `duration` - Fade duration in seconds
    /// * `curve` - Fade curve type
    ///
    /// # Examples
    /// ```python
    /// # Linear fade-out over 1 second
    /// audio.fade_out_(duration=1.0, curve='linear')
    ///
    /// # Logarithmic fade-out
    /// audio.fade_out_(duration=0.5, curve='logarithmic')
    /// ```
    pub(crate) fn fade_out_inplace_impl(&mut self, duration: f64, curve: &str) -> PyResult<()> {
        let fade_curve = Self::parse_fade_curve_impl(curve)?;
        self.mutate_inner(|inner| inner.fade_out(duration, fade_curve))
            .map_err(map_error)
    }

    /// Apply fade-out envelope (functional version).
    ///
    /// # Arguments
    /// * `duration` - Fade duration in seconds
    /// * `curve` - Fade curve type
    ///
    /// # Returns
    /// New AudioSamples object with fade-out applied
    ///
    /// # Examples
    /// ```python
    /// # Create faded copy (original unchanged)
    /// faded = audio.fade_out(duration=1.0, curve='smoothstep')
    /// ```
    pub(crate) fn fade_out_impl(&self, duration: f64, curve: &str) -> PyResult<PyAudioSamples> {
        let mut result = self.copy();
        result.fade_out_inplace_impl(duration, curve)?;
        Ok(result)
    }

    // ==========================================
    // STATIC METHODS FOR COMBINING OPERATIONS
    // ==========================================

    /// Concatenate multiple audio segments into one.
    ///
    /// All segments must have the same sample rate and number of channels.
    ///
    /// # Arguments
    /// * `segments` - List of AudioSamples objects to concatenate
    /// * `axis` - Concatenation axis (0 for time, 1 for channels - currently only 0 supported)
    ///
    /// # Returns
    /// New AudioSamples object containing all segments joined together
    ///
    /// # Examples
    /// ```python
    /// # Join multiple clips
    /// combined = aus.AudioSamples.concatenate([clip1, clip2, clip3])
    ///
    /// # Join with intro and outro
    /// full_track = aus.AudioSamples.concatenate([intro, main_audio, outro])
    /// ```
    pub(crate) fn concatenate_impl(
        segments: Vec<&PyAudioSamples>,
        axis: i32,
    ) -> PyResult<PyAudioSamples> {
        if segments.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Cannot concatenate empty list of segments",
            ));
        }

        if axis != 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Only axis=0 (time concatenation) is currently supported",
            ));
        }

        // Convert to inner AudioSamples for concatenation
        let inner_segments: Vec<crate::AudioSamples<f64>> =
            segments.iter().map(|seg| seg.as_f64().unwrap()).collect();

        let concatenated =
            crate::AudioSamples::concatenate(&inner_segments[..]).map_err(map_error)?;
        Ok(PyAudioSamples::from_inner(concatenated))
    }

    /// Mix multiple audio sources together.
    ///
    /// All sources must have the same sample rate, number of channels, and length.
    ///
    /// # Arguments
    /// * `sources` - List of AudioSamples objects to mix
    /// * `weights` - Optional list of mixing weights (default: equal weights)
    /// * `normalize` - Whether to normalize the result to prevent clipping
    ///
    /// # Returns
    /// New AudioSamples object containing the mixed audio
    ///
    /// # Examples
    /// ```python
    /// # Simple equal mix
    /// mixed = aus.AudioSamples.mix([track1, track2])
    ///
    /// # Weighted mix (track1 at 70%, track2 at 30%)
    /// weighted_mix = aus.AudioSamples.mix([track1, track2], weights=[0.7, 0.3])
    ///
    /// # Mix with automatic normalization
    /// normalized_mix = aus.AudioSamples.mix([loud1, loud2], normalize=True)
    /// ```
    pub(crate) fn mix_impl(
        sources: Vec<&PyAudioSamples>,
        weights: Option<Vec<f64>>,
        normalize: bool,
    ) -> PyResult<PyAudioSamples> {
        if sources.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Cannot mix empty list of sources",
            ));
        }

        // Validate weights if provided
        if let Some(ref w) = weights {
            if w.len() != sources.len() {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Number of weights must match number of sources",
                ));
            }
        }

        // Convert to inner AudioSamples for mixing
        let inner_sources: Vec<crate::AudioSamples<f64>> =
            sources.iter().map(|src| src.as_f64().unwrap()).collect();

        let weights_slice = weights.as_ref().map(|w| w.as_slice());
        let mixed =
            crate::AudioSamples::mix(&inner_sources[..], weights_slice).map_err(map_error)?;

        let result = if normalize {
            // Apply peak normalization to prevent clipping
            let mut normalized = mixed;
            normalized
                .normalize(-1.0, 1.0, crate::operations::NormalizationMethod::Peak)
                .map_err(map_error)?;
            normalized
        } else {
            mixed
        };

        Ok(PyAudioSamples::from_inner(result))
    }

    // ==========================================
    // PERTURBATION OPERATIONS (data augmentation)
    // ==========================================

    /// Apply perturbation to audio samples for data augmentation (in-place).
    ///
    /// # Arguments
    /// * `method` - Perturbation method ('gaussian_noise', 'random_gain', 'high_pass_filter', 'pitch_shift')
    /// * `seed` - Optional random seed for deterministic results
    /// * `target_snr_db` - Target SNR in dB (for gaussian_noise, default: 20.0)
    /// * `noise_color` - Noise color ('white', 'pink', 'brown', default: 'white')
    /// * `min_gain_db` - Minimum gain in dB (for random_gain, default: -3.0)
    /// * `max_gain_db` - Maximum gain in dB (for random_gain, default: 3.0)
    /// * `cutoff_hz` - Cutoff frequency in Hz (for high_pass_filter, default: 80.0)
    /// * `slope_db_per_octave` - Filter slope (for high_pass_filter, optional)
    /// * `semitones` - Pitch shift in semitones (for pitch_shift, default: 0.0)
    /// * `preserve_formants` - Whether to preserve formants (for pitch_shift, default: False)
    ///
    /// # Examples
    /// ```python
    /// # Add white noise at 20dB SNR with deterministic seed
    /// audio.perturb_('gaussian_noise', seed=12345, target_snr_db=20.0)
    ///
    /// # Apply random gain between -2dB and +2dB
    /// audio.perturb_('random_gain', min_gain_db=-2.0, max_gain_db=2.0)
    ///
    /// # High-pass filter to remove rumble
    /// audio.perturb_('high_pass_filter', cutoff_hz=80.0)
    ///
    /// # Pitch shift up by 2 semitones
    /// audio.perturb_('pitch_shift', semitones=2.0)
    /// ```
    pub(crate) fn perturb_inplace_impl(
        &mut self,
        method: &str,
        seed: Option<u64>,
        target_snr_db: Option<f64>,
        noise_color: Option<&str>,
        min_gain_db: Option<f64>,
        max_gain_db: Option<f64>,
        cutoff_hz: Option<f64>,
        slope_db_per_octave: Option<f64>,
        semitones: Option<f64>,
        preserve_formants: Option<bool>,
    ) -> PyResult<()> {
        let perturbation_method = Self::parse_perturbation_method_impl(
            method,
            target_snr_db,
            noise_color,
            min_gain_db,
            max_gain_db,
            cutoff_hz,
            slope_db_per_octave,
            semitones,
            preserve_formants,
        )?;

        let config = if let Some(seed_val) = seed {
            PerturbationConfig::with_seed(perturbation_method, seed_val)
        } else {
            PerturbationConfig::new(perturbation_method)
        };

        self.mutate_inner(|inner| inner.perturb_(&config))
            .map_err(map_error)?;
        Ok(())
    }

    /// Apply perturbation to audio samples for data augmentation (functional).
    ///
    /// Returns a new AudioSamples object with perturbation applied.
    /// 
    /// # Arguments
    /// Same as `perturb_` method
    ///
    /// # Returns
    /// New AudioSamples object with perturbation applied
    ///
    /// # Examples
    /// ```python
    /// # Create perturbed copy with pink noise
    /// noisy = audio.perturb('gaussian_noise', noise_color='pink', target_snr_db=15.0)
    ///
    /// # Apply deterministic random gain
    /// gained = audio.perturb('random_gain', seed=42, min_gain_db=-1.0, max_gain_db=1.0)
    /// ```
    pub(crate) fn perturb_impl(
        &self,
        method: &str,
        seed: Option<u64>,
        target_snr_db: Option<f64>,
        noise_color: Option<&str>,
        min_gain_db: Option<f64>,
        max_gain_db: Option<f64>,
        cutoff_hz: Option<f64>,
        slope_db_per_octave: Option<f64>,
        semitones: Option<f64>,
        preserve_formants: Option<bool>,
    ) -> PyResult<PyAudioSamples> {
        let mut result = self.copy();
        result.perturb_inplace_impl(
            method,
            seed,
            target_snr_db,
            noise_color,
            min_gain_db,
            max_gain_db,
            cutoff_hz,
            slope_db_per_octave,
            semitones,
            preserve_formants,
        )?;
        Ok(result)
    }

    /// Cross-fade between two audio sources.
    ///
    /// Creates a smooth transition from the first audio to the second.
    ///
    /// # Arguments
    /// * `audio1` - First audio source
    /// * `audio2` - Second audio source
    /// * `duration` - Cross-fade duration in seconds
    /// * `curve` - Fade curve type
    /// * `offset` - Offset into audio2 where cross-fade begins (default: 0.0)
    ///
    /// # Returns
    /// New AudioSamples object with cross-faded audio
    ///
    /// # Examples
    /// ```python
    /// # Simple cross-fade between two tracks
    /// crossfaded = aus.AudioSamples.crossfade(track1, track2, duration=2.0)
    ///
    /// # Smooth cross-fade with custom curve
    /// smooth_fade = aus.AudioSamples.crossfade(
    ///     track1, track2, duration=3.0, curve='smoothstep'
    /// )
    /// ```
    pub(crate) fn crossfade_impl(
        audio1: &PyAudioSamples,
        audio2: &PyAudioSamples,
        duration: f64,
        curve: &str,
        offset: f64,
    ) -> PyResult<PyAudioSamples> {
        // Check compatibility
        if audio1.sample_rate() != audio2.sample_rate() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Sample rates must match for cross-fade",
            ));
        }

        if audio1.channels() != audio2.channels() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Channel counts must match for cross-fade",
            ));
        }

        let fade_curve = PyAudioSamples::parse_fade_curve_impl(curve)?;

        // Create fade-out copy of audio1 (last 'duration' seconds)
        let audio1_f64 = audio1.as_f64().map_err(map_error)?;
        let audio1_duration = audio1_f64.duration_seconds();
        let fade_start = (audio1_duration - duration).max(0.0);
        let mut audio1_faded = audio1_f64
            .trim(fade_start, audio1_duration)
            .map_err(map_error)?;
        audio1_faded
            .fade_out(duration, fade_curve.clone())
            .map_err(map_error)?;

        // Create fade-in copy of audio2 (first 'duration' seconds, starting at offset)
        let audio2_f64 = audio2.as_f64().map_err(map_error)?;
        let fade_end = offset + duration;
        let mut audio2_faded = audio2_f64.trim(offset, fade_end).map_err(map_error)?;
        audio2_faded
            .fade_in(duration, fade_curve)
            .map_err(map_error)?;

        // Mix the faded portions
        let faded_sources = vec![audio1_faded.clone(), audio2_faded.clone()];
        let crossfade_section =
            crate::AudioSamples::mix(&faded_sources[..], None).map_err(map_error)?;

        // Create the final result by concatenating:
        // [audio1_start] + [crossfade_section] + [audio2_remainder]
        let mut final_segments = Vec::new();

        // Add beginning of audio1 (before fade)
        if fade_start > 0.0 {
            let audio1_start = audio1_f64.trim(0.0, fade_start).map_err(map_error)?;
            final_segments.push(audio1_start);
        }

        // Add crossfade section
        final_segments.push(crossfade_section);

        // Add remainder of audio2 (after fade)
        if fade_end < audio2_f64.duration_seconds() {
            let audio2_remainder = audio2_f64
                .trim(fade_end, audio2_f64.duration_seconds())
                .map_err(map_error)?;
            final_segments.push(audio2_remainder);
        }

        // Concatenate all segments
        let result = crate::AudioSamples::concatenate(&final_segments[..]).map_err(map_error)?;

        Ok(PyAudioSamples::from_inner(result))
    }
}
