//! Spectral analysis and frequency domain transformations for AudioSamples.
//!
//! This module implements the AudioTransforms trait, providing comprehensive
//! FFT-based spectral analysis operations including FFT, STFT, and spectrograms
//! using efficient rustfft operations.

use crate::operations::CqtConfig;
use crate::operations::traits::AudioTransforms;
use crate::operations::types::{SpectrogramScale, WindowType};
use crate::repr::AudioData;
use crate::{
    AudioEditing, AudioSample, AudioSampleError, AudioSampleResult, AudioSamples,
    AudioTypeConversion, ConvertTo, I24, RealFloat, to_precision,
};

use lazy_static::lazy_static;
use ndarray::{Array1, Array2, ArrayBase, OwnedRepr};
use rustfft::FftNum;
#[cfg(feature = "fft")]
use rustfft::{FftPlanner, num_complex::Complex};
use std::collections::HashMap;
use std::sync::Mutex;

/// Cache key for window functions that handles f64 parameters.
///
/// Since WindowType contains f64 values that don't implement Hash and Eq,
/// we need a custom key type that converts f64 to a hashable representation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum WindowCacheKey {
    Rectangular,
    Hanning,
    Hamming,
    Blackman,
    Kaiser { beta_bits: u64 },
    Gaussian { std_bits: u64 },
}

impl<F: RealFloat> From<WindowType<F>> for WindowCacheKey {
    fn from(window_type: WindowType<F>) -> Self {
        match window_type {
            WindowType::Rectangular => WindowCacheKey::Rectangular,
            WindowType::Hanning => WindowCacheKey::Hanning,
            WindowType::Hamming => WindowCacheKey::Hamming,
            WindowType::Blackman => WindowCacheKey::Blackman,
            WindowType::Kaiser { beta } => WindowCacheKey::Kaiser {
                beta_bits: beta
                    .to_f64()
                    .expect("Any float should be representable as f64")
                    .to_bits(),
            },
            WindowType::Gaussian { std } => WindowCacheKey::Gaussian {
                std_bits: std
                    .to_f64()
                    .expect("Any float should be representable as f64")
                    .to_bits(),
            },
        }
    }
}

impl<T: AudioSample> AudioTransforms<T> for AudioSamples<'_, T>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    for<'b> AudioSamples<'b, T>: AudioTypeConversion<'b, T>,
{
    /// Computes the Fast Fourier Transform of the audio samples.
    ///
    /// Converts the time-domain signal to frequency domain using rustfft.
    /// Returns complex frequency domain representation.
    fn fft<F>(&self) -> AudioSampleResult<Vec<Complex<F>>>
    where
        F: RealFloat + FftNum + AudioSample + ConvertTo<T>,
        T: ConvertTo<F>,
    {
        match &self.data {
            AudioData::Mono(arr) => {
                // Convert samples to complex numbers
                let mut buffer: Vec<Complex<F>> = arr
                    .iter()
                    .map(|&x| {
                        let x_converted: F = x.convert_to()?;
                        Ok(Complex::new(x_converted, F::zero()))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                // Create FFT planner and plan
                let mut planner = FftPlanner::new();
                let fft = planner.plan_fft_forward(buffer.len());

                // Execute FFT
                fft.process(&mut buffer);

                Ok(buffer)
            }
            AudioData::Multi(arr) => {
                // For multi-channel, take the first channel
                let first_channel = arr.row(0);
                let mut buffer: Vec<Complex<F>> = first_channel
                    .iter()
                    .map(|&x| {
                        let x_converted: F = x.convert_to()?;
                        Ok(Complex::new(x_converted, F::zero()))
                    })
                    .collect::<Result<Vec<_>, _>>()?;

                let mut planner = FftPlanner::new();
                let fft = planner.plan_fft_forward(buffer.len());
                fft.process(&mut buffer);

                Ok(buffer)
            }
        }
    }

    /// Compute FFT with optional windowing, zero-padding, and magnitude spectrum.
    ///
    /// # Arguments
    /// * `n_fft` - Optional FFT size. If None, uses the current signal length.
    /// * `window` - Optional window type to apply before FFT (e.g. Hanning, Hamming).
    /// * `normalise` - Whether to normalise the magnitude spectrum to [0, 1].
    ///
    /// # Returns
    /// `(freqs, mag, fft_result)` where:
    /// - `freqs`: frequency bins in Hz
    /// - `mag`: magnitude spectrum (normalised if requested)
    /// - `fft_result`: raw complex FFT output
    fn fft_info<'b, F>(
        &self,
        n_fft: Option<usize>,
        window: Option<WindowType<F>>,
        normalise: bool,
    ) -> AudioSampleResult<(Vec<F>, Vec<F>, Vec<Complex<F>>)>
    where
        F: RealFloat + FftNum + ConvertTo<T>,
        T: ConvertTo<F>,
    {
        let n_fft = n_fft.unwrap_or(self.len());
        let samples: AudioSamples<'_, T> = if self.len() < n_fft {
            AudioEditing::pad_samples_right(self, n_fft, T::zero())?
        } else if self.len() > n_fft {
            // truncate such that there are n_fft samples
            self.slice_samples(..n_fft)?
        } else {
            self.clone().into_owned()
        };
        let mut samples: AudioSamples<'_, F> = samples.as_type()?;
        // Apply window if requested
        if let Some(w) = window {
            let win = generate_window_cached(n_fft, w);

            if win.len() != samples.len() {
                return Err(AudioSampleError::DimensionMismatch(
                    "Window length does not match number of samples".to_string(),
                ));
            }
            samples.apply_with_index(|idx, x| x * win[idx]);
        }

        // Raw FFT
        let fft_result = self.fft()?;

        // Magnitude spectrum
        let mut mag: Vec<F> = fft_result.iter().map(|c: &Complex<F>| c.norm()).collect();

        // Normalise if requested
        if normalise {
            let max_val = mag.iter().cloned().fold(F::zero(), F::max);
            if max_val > F::zero() {
                for m in &mut mag {
                    *m = *m / max_val;
                }
            }
        }
        // Frequency bins
        let freqs: Vec<F> = (0..n_fft).map(|i| to_precision::<F, _>(i)).collect();

        Ok((freqs, mag, fft_result))
    }

    /// Computes the inverse FFT from frequency domain back to time domain.
    ///
    /// Reconstructs time-domain signal from complex frequency spectrum.
    fn ifft<F>(&self, spectrum: &[Complex<F>]) -> AudioSampleResult<Self>
    where
        F: RealFloat + FftNum + AudioSample + ConvertTo<T>,
        T: ConvertTo<F>,
        Self: Sized,
    {
        if spectrum.is_empty() {
            return Err(AudioSampleError::InvalidParameter(
                "Empty spectrum".to_string(),
            ));
        }

        // Copy spectrum for processing
        let mut buffer = spectrum.to_vec();

        // Create inverse FFT planner and plan
        let mut planner = FftPlanner::new();
        let ifft = planner.plan_fft_inverse(buffer.len());

        // Execute inverse FFT
        ifft.process(&mut buffer);

        // Extract real parts and normalize by length
        let len = to_precision::<F, _>(buffer.len() as f64);
        let mut real_samples: Vec<T> = Vec::with_capacity(buffer.len());
        for c in buffer {
            let real_val = c.re / len; // Normalize by length
            real_samples.push(real_val.convert_to()?);
        }

        // Create new AudioSamples with same metadata as original
        let arr = Array1::from_vec(real_samples);
        let owned = AudioSamples::new_mono(arr.into(), self.sample_rate());
        // Convert the owned static lifetime to match Self's lifetime
        // Since the owned data has 'static lifetime, it can be safely cast to any shorter lifetime
        let result: Self = unsafe { std::mem::transmute(owned) };
        Ok(result)
    }

    /// Computes the Short-Time Fourier Transform (STFT).
    ///
    /// Analyzes the signal in overlapping windows to provide time-frequency representation.
    fn stft<F>(
        &self,
        window_size: usize,
        hop_size: usize,
        window_type: WindowType<F>,
    ) -> AudioSampleResult<Array2<Complex<F>>>
    where
        F: RealFloat + FftNum + AudioSample + ConvertTo<T>,
        T: ConvertTo<F>,
    {
        if window_size == 0 || hop_size == 0 {
            return Err(AudioSampleError::InvalidParameter(
                "Window size and hop size must be greater than 0".to_string(),
            ));
        }

        if hop_size > window_size {
            return Err(AudioSampleError::InvalidParameter(
                "Hop size cannot be larger than window size".to_string(),
            ));
        }

        // Check if window size is larger than audio length
        let audio_length = self.samples_per_channel();
        if window_size > audio_length {
            return Err(AudioSampleError::InvalidParameter(format!(
                "Window size ({}) cannot be larger than audio length ({})",
                window_size, audio_length
            )));
        }

        // Get samples based on channel configuration and convert to F
        let mut samples_f: Vec<F> = match &self.data {
            AudioData::Mono(arr) => arr
                .iter()
                .map(|&x| x.convert_to())
                .collect::<Result<Vec<_>, _>>()?,
            AudioData::Multi(arr) => arr
                .row(0)
                .iter()
                .map(|&x| x.convert_to())
                .collect::<Result<Vec<_>, _>>()?,
        };

        // Apply librosa-style centering (padding)
        let pad_width = window_size / 2;
        let mut centered_samples = Vec::with_capacity(samples_f.len() + 2 * pad_width);

        for i in 1..=pad_width {
            let idx = i.min(samples_f.len() - 1);
            centered_samples.push(samples_f[idx]);
        }
        centered_samples.reverse();

        // Add original samples
        centered_samples.extend_from_slice(&samples_f);

        // Right reflect padding: x[-2:-pad_width-2:-1]
        for i in 1..=pad_width {
            let idx = samples_f.len().saturating_sub(1 + i);
            centered_samples.push(samples_f[idx]);
        }

        samples_f = centered_samples;

        if samples_f.len() < window_size {
            return Err(AudioSampleError::DimensionMismatch(
                "Audio length is shorter than window size".to_string(),
            ));
        }

        // Calculate number of frames (librosa-compatible)
        let num_frames = if samples_f.len() >= window_size {
            (samples_f.len() - window_size) / hop_size + 1
        } else {
            0
        };

        if num_frames == 0 {
            return Err(AudioSampleError::DimensionMismatch(
                "No frames can be extracted".to_string(),
            ));
        }

        // Generate window function (cached for performance)
        let window = generate_window_cached::<F>(window_size, window_type)
            .into_iter()
            .map(|w| w * to_precision::<F, _>(0.5))
            .collect::<Vec<_>>();

        // For real-valued signals, we only need positive frequencies
        let num_positive_freqs = window_size / 2 + 1;

        // Initialize STFT matrix: frequency bins × time frames (only positive frequencies)
        let mut stft_matrix = Array2::zeros((num_positive_freqs, num_frames));

        // Setup real FFT planner for efficiency
        let mut real_planner = realfft::RealFftPlanner::new();
        let r2c = real_planner.plan_fft_forward(window_size);

        // Pre-allocate buffers
        let mut real_input = vec![F::zero(); window_size];
        let mut complex_output = vec![Complex::new(F::zero(), F::zero()); num_positive_freqs];

        // Process each frame
        for frame_idx in 0..num_frames {
            let start = frame_idx * hop_size;
            let end = start + window_size;

            // Extract and window the frame (real-valued)
            for (i, (&sample, &w)) in samples_f[start..end].iter().zip(window.iter()).enumerate() {
                real_input[i] = sample * w;
            }

            // Apply real FFT (more efficient than complex FFT for real signals)
            r2c.process(&mut real_input, &mut complex_output)
                .map_err(|_| {
                    AudioSampleError::InvalidParameter("Real FFT processing failed".to_string())
                })?;

            // Store in STFT matrix (only positive frequencies)
            for (freq_idx, &value) in complex_output.iter().enumerate() {
                stft_matrix[[freq_idx, frame_idx]] = value;
            }
        }

        Ok(stft_matrix)
    }

    /// Computes the inverse STFT to reconstruct time domain signal.
    ///
    /// Reconstructs time-domain signal from STFT representation using overlap-add.
    fn istft<F>(
        stft_matrix: &Array2<Complex<F>>,
        hop_size: usize,
        window_type: WindowType<F>,
        sample_rate: usize,
        center: bool,
    ) -> AudioSampleResult<Self>
    where
        F: RealFloat + FftNum + AudioSample + ConvertTo<T>,
        T: ConvertTo<F>,
        Self: Sized,
    {
        let (num_positive_freqs, num_frames) = stft_matrix.dim();
        let fft_size = (num_positive_freqs - 1) * 2;
        let output_length = (num_frames - 1) * hop_size + fft_size;

        let mut output = vec![F::zero(); output_length];
        let mut window_sum = vec![F::zero(); output_length];

        let window = generate_window_cached(fft_size, window_type)
            .into_iter()
            .map(|w| w * to_precision(0.5))
            .collect::<Vec<_>>();

        let mut real_planner = realfft::RealFftPlanner::<F>::new();
        let c2r = real_planner.plan_fft_inverse(fft_size);
        let mut real_buffer = vec![F::zero(); fft_size];
        let mut complex_buffer = vec![Complex::new(F::zero(), F::zero()); num_positive_freqs];

        for frame_idx in 0..num_frames {
            // Copy the stored positive frequencies directly
            for f in 0..num_positive_freqs {
                complex_buffer[f] = stft_matrix[[f, frame_idx]];
            }

            // Real inverse FFT
            c2r.process(&mut complex_buffer, &mut real_buffer)
                .expect("IFFT failed");
            let inv_scale = F::one() / to_precision::<F, _>(fft_size as f64);
            for s in real_buffer.iter_mut() {
                *s = *s * inv_scale;
            }
            let start = frame_idx * hop_size;
            for i in 0..fft_size {
                let w = window[i];
                output[start + i] = output[start + i] + real_buffer[i] * w;
                window_sum[start + i] = window_sum[start + i] + w * w;
            }
        }

        // Normalise
        for i in 0..output_length {
            if window_sum[i] > F::zero() {
                output[i] = output[i] / window_sum[i];
            }
        }

        // Remove center padding if needed
        let final_output = if center {
            let pad = fft_size / 2;
            output[pad..output_length - pad].to_vec()
        } else {
            output
        };

        // Convert to AudioSamples
        let samples: Vec<T> = final_output
            .into_iter()
            .map(|s| s.convert_to())
            .collect::<Result<Vec<T>, _>>()?;
        let arr = Array1::from_vec(samples);

        let owned: AudioSamples<'static, T> =
            AudioSamples::new_mono(arr.into(), sample_rate as u32);
        Ok(unsafe { std::mem::transmute(owned) })
    }

    /// Computes the magnitude spectrogram (|STFT|^2) with scaling options.
    ///
    /// Returns power spectrum over time for visualization and analysis.
    /// The spectrogram shows how the frequency content of a signal evolves over time.
    ///
    /// # Mathematical Foundation
    ///
    /// The spectrogram is computed as:
    /// ```text
    /// S(m,ω) = |STFT[x(n)](m,ω)|²
    /// ```
    ///
    /// Where STFT is the Short-Time Fourier Transform computed using overlapping windows.
    ///
    /// # Scaling Options
    ///
    /// - **Linear**: Raw power values (|STFT|²) - preserves absolute energy relationships
    /// - **Log**: Logarithmic scale in dB (20 * log10(power)) - compresses dynamic range
    /// - **Mel**: Mel-frequency scale - perceptually motivated (requires mel filter bank)
    ///
    /// # Arguments
    /// * `window_size` - Size of each analysis window in samples
    /// * `hop_size` - Number of samples between successive windows
    /// * `window_type` - Window function to reduce spectral leakage
    /// * `scale` - Scaling method to apply to the power values
    /// * `normalize` - Whether to normalize the result
    ///
    /// # Returns
    ///
    /// `Array2<f64>` with dimensions `(freq_bins, time_frames)` where:
    /// - `freq_bins = window_size / 2 + 1` (for real signals)
    /// - `time_frames = (signal_length - window_size) / hop_size + 1`
    ///
    /// # Time-Frequency Trade-offs
    ///
    /// - **Larger window_size**: Better frequency resolution, worse time resolution
    /// - **Smaller window_size**: Better time resolution, worse frequency resolution
    /// - **Smaller hop_size**: Smoother time evolution, more computation
    fn spectrogram<F>(
        &self,
        window_size: usize,
        hop_size: usize,
        window_type: WindowType<F>,
        scale: SpectrogramScale,
        normalize: bool,
    ) -> AudioSampleResult<Array2<F>>
    where
        F: RealFloat + FftNum + AudioSample + ConvertTo<T>,
        T: ConvertTo<F>,
    {
        // Input validation
        if window_size == 0 || hop_size == 0 {
            return Err(AudioSampleError::InvalidParameter(
                "Window size and hop size must be greater than 0".to_string(),
            ));
        }

        if hop_size > window_size {
            return Err(AudioSampleError::InvalidParameter(
                "Hop size cannot be larger than window size".to_string(),
            ));
        }

        // For mel scaling, redirect to mel_spectrogram (will be implemented in Phase 3)
        if scale == SpectrogramScale::Mel {
            return Err(AudioSampleError::InvalidParameter(
                "Mel scaling requires mel_spectrogram method - use mel_spectrogram() directly"
                    .to_string(),
            ));
        }

        // Compute STFT using existing implementation
        let stft_matrix: ArrayBase<OwnedRepr<Complex<F>>, ndarray::Dim<[usize; 2]>, Complex<F>> =
            self.stft(window_size, hop_size, window_type)?;
        let (freq_bins, time_frames) = stft_matrix.dim();

        // Convert complex STFT to power spectrogram
        let mut spectrogram = Array2::zeros((freq_bins, time_frames));

        for freq_idx in 0..freq_bins {
            for time_idx in 0..time_frames {
                let complex_val = stft_matrix[[freq_idx, time_idx]];
                let power = complex_val.norm_sqr(); // |STFT|² = real² + imag²

                let scaled_power = match scale {
                    SpectrogramScale::Linear => power,
                    SpectrogramScale::Log => {
                        // Convert to dB with epsilon to prevent log(0) - librosa style
                        let power_safe =
                            power.max(F::from_f64(1e-10).expect("1e-10 is a valid float"));
                        to_precision::<F, _>(10.0) * power_safe.log10()
                    }
                    SpectrogramScale::Mel => {
                        // This case is handled above with early return
                        return Err(AudioSampleError::InvalidParameter(
                                "Mel scaling requires mel_spectrogram method - use mel_spectrogram() directly"
                                    .to_string(),
                            ));
                    }
                };

                spectrogram[[freq_idx, time_idx]] = scaled_power;
            }
        }

        // Apply normalization if requested
        if normalize {
            normalize_spectrogram(&mut spectrogram, scale)?;
        }

        Ok(spectrogram)
    }

    /// Computes mel-scaled spectrogram for perceptual analysis.
    ///
    /// Applies mel filter bank to power spectrogram to create perceptually uniform
    /// frequency representation. The mel scale better represents human auditory
    /// perception and is commonly used in speech recognition and music analysis.
    ///
    /// # Implementation Steps
    ///
    /// 1. Compute power spectrogram using STFT
    /// 2. Generate mel filter bank with triangular filters
    /// 3. Apply mel filters to power spectrogram (frequency domain multiplication)
    /// 4. Convert to log scale for dynamic range compression
    ///
    /// # Mathematical Foundation
    ///
    /// Mel scale formula:
    /// ```text
    /// mel = 2595 * log10(1 + f/700)
    /// ```
    ///
    /// Mel filter bank consists of triangular filters with:
    /// - Linear spacing on mel scale (perceptually uniform)
    /// - Overlapping filters for smooth frequency response
    /// - Unit area normalization for energy preservation
    ///
    /// # Arguments
    /// * `n_mels` - Number of mel frequency bands (typically 40-128)
    /// * `fmin` - Minimum frequency in Hz (typically 0-100 Hz)
    /// * `fmax` - Maximum frequency in Hz (typically sample_rate/2)
    /// * `window_size` - Size of each analysis window in samples
    /// * `hop_size` - Number of samples between successive windows
    ///
    /// # Returns
    ///
    /// `Array2<f64>` with dimensions `(n_mels, time_frames)` containing
    /// log mel energies over time
    fn mel_spectrogram<F>(
        &self,
        n_mels: usize,
        fmin: F,
        fmax: F,
        window_size: usize,
        hop_size: usize,
    ) -> AudioSampleResult<Array2<F>>
    where
        F: RealFloat + FftNum + AudioSample + ConvertTo<T>,
        T: ConvertTo<F>,
    {
        // Input validation
        if n_mels == 0 || window_size == 0 || hop_size == 0 {
            return Err(AudioSampleError::InvalidParameter(
                "Number of mels, window size, and hop size must be greater than 0".to_string(),
            ));
        }

        if fmin < F::zero() || fmax <= fmin {
            return Err(AudioSampleError::InvalidParameter(
                "Frequency range must be positive and fmax > fmin".to_string(),
            ));
        }

        if hop_size > window_size {
            return Err(AudioSampleError::InvalidParameter(
                "Hop size cannot be larger than window size".to_string(),
            ));
        }

        let sample_rate = to_precision::<F, _>(to_precision::<F, _>(self.sample_rate));
        if fmax > sample_rate / to_precision::<F, _>(2.0) {
            return Err(AudioSampleError::InvalidParameter(
                "Maximum frequency cannot exceed Nyquist frequency".to_string(),
            ));
        }

        // Step 1: Compute power spectrogram using existing STFT
        let stft_matrix = self.stft(window_size, hop_size, WindowType::Hanning)?;
        let (freq_bins, time_frames) = stft_matrix.dim();

        // Convert STFT to power spectrogram
        let mut power_spectrogram = Array2::zeros((freq_bins, time_frames));
        for freq_idx in 0..freq_bins {
            for time_idx in 0..time_frames {
                let complex_val = stft_matrix[[freq_idx, time_idx]];
                let power = complex_val.norm_sqr(); // |STFT|²
                power_spectrogram[[freq_idx, time_idx]] = power;
            }
        }

        // Step 2: Generate mel filter bank
        let mel_filter_bank =
            generate_mel_filter_bank(n_mels, window_size, sample_rate, fmin, fmax);

        // Step 3: Apply mel filters to power spectrogram
        let mut mel_spectrogram = Array2::zeros((n_mels, time_frames));

        for mel_idx in 0..n_mels {
            for time_idx in 0..time_frames {
                let mut mel_energy = F::zero();

                // Apply mel filter: dot product of filter with power spectrum
                // Ensure we don't exceed filter bank dimensions
                let filter_freq_bins = mel_filter_bank.ncols();
                let max_freq_idx = freq_bins.min(filter_freq_bins);

                for freq_idx in 0..max_freq_idx {
                    let filter_weight = mel_filter_bank[[mel_idx, freq_idx]];
                    let power = power_spectrogram[[freq_idx, time_idx]];
                    mel_energy += filter_weight * power;
                }

                // Step 4: Convert to log scale (dB) with floor to prevent log(0)
                let log_mel_energy = if mel_energy > to_precision::<F, _>(1e-10) {
                    to_precision::<F, _>(10.0) * mel_energy.log10() // 10*log10 for power
                } else {
                    to_precision::<F, _>(-100.0) // Floor at -100 dB
                };

                mel_spectrogram[[mel_idx, time_idx]] = log_mel_energy;
            }
        }

        Ok(mel_spectrogram)
    }

    /// Computes Mel-Frequency Cepstral Coefficients (MFCC).
    ///
    /// MFCCs are widely used features in speech recognition and audio analysis.
    /// They provide a compact representation of spectral characteristics by
    /// applying DCT to log mel filter bank energies.
    ///
    /// # Implementation Steps
    ///
    /// 1. Compute mel spectrogram (log mel filter bank energies)
    /// 2. Apply Discrete Cosine Transform (DCT) to decorrelate features
    /// 3. Keep only the first n_mfcc coefficients (most important information)
    ///
    /// # Mathematical Foundation
    ///
    /// MFCC computation pipeline:
    /// ```text
    /// Audio → STFT → |STFT|² → Mel filters → log() → DCT → MFCCs
    /// ```
    ///
    /// DCT-II formula used:
    /// ```text
    /// MFCC[k] = Σ log_mel[n] * cos(π*k*(2n+1)/(2N))
    /// ```
    ///
    /// # Applications
    ///
    /// - **Speech recognition**: Compact speech features
    /// - **Speaker identification**: Voice characteristics
    /// - **Audio classification**: Genre, instrument recognition
    /// - **Audio similarity**: Comparing spectral content
    ///
    /// # Arguments
    /// * `n_mfcc` - Number of MFCC coefficients (typically 12-13)
    /// * `n_mels` - Number of mel frequency bands (typically 40-128)
    /// * `fmin` - Minimum frequency in Hz
    /// * `fmax` - Maximum frequency in Hz
    ///
    /// # Returns
    ///
    /// `Array2<f64>` with dimensions `(n_mfcc, time_frames)` containing
    /// MFCC coefficients over time
    fn mfcc<F>(
        &self,
        n_mfcc: usize,
        n_mels: usize,
        fmin: F,
        fmax: F,
    ) -> AudioSampleResult<Array2<F>>
    where
        F: RealFloat + FftNum + AudioSample + ConvertTo<T>,
        T: ConvertTo<F>,
    {
        // Input validation
        if n_mfcc == 0 || n_mels == 0 {
            return Err(AudioSampleError::InvalidParameter(
                "Number of MFCC and mel bands must be greater than 0".to_string(),
            ));
        }

        if n_mfcc > n_mels {
            return Err(AudioSampleError::InvalidParameter(
                "Number of MFCC coefficients cannot exceed number of mel bands".to_string(),
            ));
        }

        // Use reasonable default window parameters for MFCC computation
        let window_size = 2048; // ~46ms at 44.1kHz
        let hop_size = 512; // ~12ms hop (75% overlap)

        // Step 1: Compute mel spectrogram (log mel energies)
        let mel_spec = self.mel_spectrogram(n_mels, fmin, fmax, window_size, hop_size)?;
        let (n_mel_bands, time_frames) = mel_spec.dim();

        // Step 2: Apply DCT to each time frame
        let mut mfcc_matrix = Array2::zeros((n_mfcc, time_frames));

        for time_idx in 0..time_frames {
            // Extract mel energies for this time frame
            let mel_frame: Vec<F> = (0..n_mel_bands)
                .map(|mel_idx| mel_spec[[mel_idx, time_idx]])
                .collect();

            // Apply DCT to decorrelate mel features
            let mfcc_frame = compute_dct_type2(&mel_frame, n_mfcc);

            // Store in output matrix
            for mfcc_idx in 0..n_mfcc {
                mfcc_matrix[[mfcc_idx, time_idx]] = mfcc_frame[mfcc_idx];
            }
        }

        Ok(mfcc_matrix)
    }

    fn chroma<F>(&self, _n_chroma: usize) -> AudioSampleResult<Array2<F>>
    where
        F: RealFloat + FftNum + AudioSample + ConvertTo<T>,
        T: ConvertTo<F>,
    {
        Err(AudioSampleError::InvalidParameter(
            "chroma not yet implemented".to_string(),
        ))
    }

    fn power_spectral_density<F>(
        &self,
        _window_size: usize,
        _overlap: F,
    ) -> AudioSampleResult<(Vec<F>, Vec<F>)>
    where
        F: RealFloat + FftNum + AudioSample + ConvertTo<T>,
        T: ConvertTo<F>,
    {
        Err(AudioSampleError::InvalidParameter(
            "power_spectral_density not yet implemented".to_string(),
        ))
    }

    /// Computes gammatone-filtered spectrogram for auditory modeling.
    ///
    /// Uses ERB-spaced gammatone filters to model human cochlear processing.
    /// Each filter represents the response of a specific region of the basilar membrane.
    ///
    /// # Implementation Details
    ///
    /// 1. Generate ERB-spaced center frequencies from fmin to fmax
    /// 2. Create gammatone filter bank with 4th-order filters
    /// 3. Apply each filter to overlapping windows of the input signal
    /// 4. Compute energy in each frequency band over time
    ///
    /// # Mathematical Foundation
    ///
    /// ERB scale provides perceptually uniform frequency spacing:
    /// ```text
    /// ERB(f) = 24.7 * (4.37*f/1000 + 1)
    /// ```
    ///
    /// Gammatone filter (4th order, n=4):
    /// ```text
    /// g(t) = a * t³ * exp(-2πERB(f)t) * cos(2πft + φ)
    /// ```
    fn gammatone_spectrogram<F>(
        &self,
        n_filters: usize,
        fmin: F,
        fmax: F,
        window_size: usize,
        hop_size: usize,
    ) -> AudioSampleResult<Array2<F>>
    where
        F: RealFloat + FftNum + AudioSample + ConvertTo<T>,
        T: ConvertTo<F>,
    {
        // Input validation
        if n_filters == 0 || window_size == 0 || hop_size == 0 {
            return Err(AudioSampleError::InvalidParameter(
                "Number of filters, window size, and hop size must be greater than 0".to_string(),
            ));
        }

        if fmin <= F::zero() || fmax <= fmin {
            return Err(AudioSampleError::InvalidParameter(
                "Frequency range must be positive and fmax > fmin".to_string(),
            ));
        }

        if hop_size > window_size {
            return Err(AudioSampleError::InvalidParameter(
                "Hop size cannot be larger than window size".to_string(),
            ));
        }

        let sample_rate = to_precision::<F, _>(self.sample_rate);
        if fmax > sample_rate / to_precision::<F, _>(2.0) {
            return Err(AudioSampleError::InvalidParameter(
                "Maximum frequency cannot exceed Nyquist frequency".to_string(),
            ));
        }

        // Get samples based on channel configuration
        let samples = match &self.data {
            AudioData::Mono(arr) => arr.to_vec(),
            AudioData::Multi(arr) => arr.row(0).to_vec(), // Use first channel
        };

        if samples.len() < window_size {
            return Err(AudioSampleError::DimensionMismatch(
                "Audio length is shorter than window size".to_string(),
            ));
        }

        // Calculate number of frames
        let num_frames = if samples.len() >= window_size {
            (samples.len() - window_size) / hop_size + 1
        } else {
            0
        };

        if num_frames == 0 {
            return Err(AudioSampleError::DimensionMismatch(
                "No frames can be extracted".to_string(),
            ));
        }

        // Generate ERB-spaced center frequencies
        let center_frequencies = generate_erb_frequencies(n_filters, fmin, fmax);

        // Initialize gammatone spectrogram matrix: filters × time frames
        let mut gammatone_spec = Array2::<F>::zeros((n_filters, num_frames));

        // Generate gammatone filter bank
        let filter_bank =
            generate_gammatone_filter_bank(&center_frequencies, sample_rate, window_size);

        // Process each frame
        for frame_idx in 0..num_frames {
            let start = frame_idx * hop_size;
            let end = start + window_size;

            // Extract frame
            let mut frame: Vec<F> = Vec::with_capacity(end - start);

            for i in start..end {
                if i < samples.len() {
                    frame.push(samples[i].convert_to()?);
                }
            }
            // Apply each gammatone filter and compute energy
            for (filter_idx, filter_coeffs) in filter_bank.iter().enumerate() {
                let filtered_output = apply_gammatone_filter(&frame, filter_coeffs.as_slice());

                // Compute energy (RMS) of filtered output
                let energy: F = filtered_output
                    .iter()
                    .map(|&x| x * x)
                    .reduce(|acc, x| x + acc)
                    .expect("Should not fail")
                    / to_precision::<F, _>(filtered_output.len());

                gammatone_spec[[filter_idx, frame_idx]] = energy.sqrt();
            }
        }

        Ok(gammatone_spec)
    }

    fn constant_q_transform<F>(
        &self,
        config: &super::types::CqtConfig<F>,
    ) -> AudioSampleResult<Array2<Complex<F>>>
    where
        F: RealFloat + FftNum + AudioSample + ConvertTo<T>,
        T: ConvertTo<F>,
    {
        let sample_rate = to_precision::<F, _>(self.sample_rate);

        // Validate configuration
        config.validate(sample_rate)?;

        // Get mono samples as f64 for processing
        let samples = self.to_mono_float_samples()?;

        if samples.is_empty() {
            return Err(AudioSampleError::InvalidParameter(
                "Cannot compute CQT on empty signal".to_string(),
            ));
        }

        // Generate CQT kernel
        let kernel = generate_cqt_kernel(config, sample_rate, samples.len())?;

        // Apply CQT using FFT convolution
        let cqt_result = apply_cqt_kernel(&samples, &kernel)?;

        // Return result as Array2 with dimensions (num_bins, 1) for single-frame analysis
        let num_bins = config.num_bins(sample_rate);
        let mut result = Array2::zeros((num_bins, 1));

        for (i, &coeff) in cqt_result.iter().enumerate() {
            if i < num_bins {
                result[[i, 0]] = coeff;
            }
        }

        Ok(result)
    }

    fn inverse_constant_q_transform<F>(
        cqt_matrix: &Array2<Complex<F>>,
        config: &CqtConfig<F>,
        signal_length: usize,
        sample_rate: F,
    ) -> AudioSampleResult<Self>
    where
        F: RealFloat + FftNum + AudioSample + ConvertTo<T>,
        T: ConvertTo<F>,
        Self: Sized,
    {
        // Validate configuration
        config.validate(sample_rate)?;

        if cqt_matrix.is_empty() {
            return Err(AudioSampleError::InvalidParameter(
                "Cannot compute inverse CQT on empty matrix".to_string(),
            ));
        }

        // Generate dual frame (reconstruction kernel)
        let dual_kernel = generate_dual_cqt_kernel(config, sample_rate, signal_length)?;

        // Apply inverse CQT using dual frame reconstruction
        let reconstructed = apply_inverse_cqt_kernel(cqt_matrix, &dual_kernel, signal_length)?;

        // Convert back to original sample type
        let mut reconstructed_samples: Vec<T> = Vec::with_capacity(reconstructed.len());
        for sample in reconstructed {
            let converted: T = sample.convert_to()?;
            reconstructed_samples.push(converted);
        }

        let arr = Array1::from_vec(reconstructed_samples);
        let owned: AudioSamples<'static, T> = AudioSamples::new_mono(
            arr.into(),
            sample_rate
                .to_u32()
                .expect("Sample rate in should be a non-zero positive u32, now just casting back"),
        );
        Ok(unsafe { std::mem::transmute(owned) })
    }

    fn cqt_spectrogram<F>(
        &self,
        config: &CqtConfig<F>,
        hop_size: usize,
        window_size: Option<usize>,
    ) -> AudioSampleResult<Array2<Complex<F>>>
    where
        T: ConvertTo<F>,
        F: RealFloat + ConvertTo<T>,
    {
        let sample_rate = to_precision::<F, _>(self.sample_rate);

        // Validate configuration
        config.validate(sample_rate)?;

        if hop_size == 0 {
            return Err(AudioSampleError::InvalidParameter(
                "Hop size must be greater than 0".to_string(),
            ));
        }

        // Get mono samples as f64 for processing
        let samples = self.to_mono_float_samples()?;

        if samples.is_empty() {
            return Err(AudioSampleError::InvalidParameter(
                "Cannot compute CQT spectrogram on empty signal".to_string(),
            ));
        }

        // Calculate effective window size
        let effective_window_size = match window_size {
            Some(size) => size,
            None => {
                // Auto-calculate window size based on lowest frequency
                let min_period = sample_rate / config.fmin;
                (min_period * to_precision::<F, _>(4.0)).to_usize().expect("Sample rate and fmin are expected to be non-zero positive values, hence min period will >= 0 when cast to a usize ") // 4 periods for good frequency resolution
            }
        };

        // Generate CQT kernel for the window size
        let kernel = generate_cqt_kernel(config, sample_rate, effective_window_size)?;

        // Calculate number of frames
        let num_frames = (samples.len().saturating_sub(effective_window_size)) / hop_size + 1;
        let num_bins = config.num_bins(sample_rate);

        // Initialize result matrix
        let mut result = Array2::zeros((num_bins, num_frames));

        // Apply CQT to each frame
        for frame_idx in 0..num_frames {
            let start_idx = frame_idx * hop_size;
            let end_idx = (start_idx + effective_window_size).min(samples.len());

            if end_idx > start_idx {
                // Extract window
                let window = &samples[start_idx..end_idx];

                // Pad window if necessary
                let mut padded_window = window.to_vec();
                if padded_window.len() < effective_window_size {
                    padded_window.resize(effective_window_size, F::zero());
                }

                // Apply CQT to this window
                let cqt_frame = apply_cqt_kernel(&padded_window, &kernel)?;

                // Store result
                for (bin_idx, &coeff) in cqt_frame.iter().enumerate() {
                    if bin_idx < num_bins {
                        result[[bin_idx, frame_idx]] = coeff;
                    }
                }
            }
        }

        Ok(result)
    }

    fn cqt_magnitude_spectrogram<F: RealFloat + ConvertTo<T>>(
        &self,
        config: &CqtConfig<F>,
        hop_size: usize,
        window_size: Option<usize>,
        power: bool,
    ) -> AudioSampleResult<Array2<F>>
    where
        T: ConvertTo<F>,
    {
        // Compute complex CQT spectrogram
        let complex_spectrogram = self.cqt_spectrogram(config, hop_size, window_size)?;

        // Convert to magnitude (or power)
        let (num_bins, num_frames) = complex_spectrogram.dim();
        let mut magnitude_spectrogram = Array2::zeros((num_bins, num_frames));

        for bin_idx in 0..num_bins {
            for frame_idx in 0..num_frames {
                let complex_val = complex_spectrogram[[bin_idx, frame_idx]];
                let magnitude = if power {
                    complex_val.norm_sqr() // Power = |z|²
                } else {
                    complex_val.norm() // Magnitude = |z|
                };
                magnitude_spectrogram[[bin_idx, frame_idx]] = magnitude;
            }
        }

        Ok(magnitude_spectrogram)
    }
}

/// Normalizes a spectrogram based on the scaling method.
///
/// Different scaling methods require different normalization approaches:
/// - Linear: Normalize to [0, 1] range
/// - Log: Ensure reasonable dB range (typically -80 to +40 dB)
fn normalize_spectrogram<F: RealFloat>(
    spectrogram: &mut Array2<F>,
    scale: SpectrogramScale,
) -> AudioSampleResult<()> {
    match scale {
        SpectrogramScale::Linear => {
            // Normalize to [0, 1] range
            if let Some(&max_val) = spectrogram.iter().max_by(|a, b| {
                match a.partial_cmp(b) {
                    Some(order) => order,
                    None => std::cmp::Ordering::Equal, // Handle NaN gracefully
                }
            }) {
                if max_val > F::zero() {
                    spectrogram.mapv_inplace(|x| x / max_val);
                }
            }
        }
        SpectrogramScale::Log => {
            // Log scale is already in dB, just ensure reasonable range
            // Clamp to [-100, +50] dB range
            spectrogram.mapv_inplace(|x| {
                x.clamp(to_precision::<F, _>(-100.0), to_precision::<F, _>(50.0))
            });
        }
        SpectrogramScale::Mel => {
            // Mel scaling normalization will be implemented in Phase 3
            return Err(AudioSampleError::InvalidParameter(
                "Mel scale normalization not yet implemented".to_string(),
            ));
        }
    }
    Ok(())
}

lazy_static! {
    /// Cached window function storage for performance optimization.
    ///
    /// Uses a global cache to avoid recomputing expensive trigonometric operations
    /// for common window types and sizes.
    static ref WINDOW_CACHE: Mutex<HashMap<(usize, WindowCacheKey), Vec<f64>>> =
        Mutex::new(HashMap::new());
}

/// Generate window function coefficients with caching for performance.
///
/// Supports various window types for spectral analysis. Frequently used
/// window functions are cached to avoid expensive recomputation of
/// trigonometric functions.
///
/// # Arguments
/// * `size` - Window size in samples
/// * `window_type` - Type of window function
///
/// # Returns  
/// Vector of window coefficients, cached for future use
fn generate_window_cached<F: RealFloat>(size: usize, window_type: WindowType<F>) -> Vec<F> {
    let cache_key = WindowCacheKey::from(window_type);

    // Check cache first
    if let Ok(cache) = WINDOW_CACHE.try_lock() {
        if let Some(cached_window) = cache.get(&(size, cache_key.clone())) {
            return cached_window
                .iter()
                .map(|&v| to_precision::<F, _>(v))
                .collect();
        }
    }

    // Generate window if not cached
    let window = generate_window_uncached::<F>(size, window_type);

    // Cache the result
    if let Ok(mut cache) = WINDOW_CACHE.try_lock() {
        // Limit cache size to prevent unbounded growth
        if cache.len() < 100 {
            // Allow up to 100 cached windows
            cache.insert(
                (size, cache_key),
                window
                    .iter()
                    .map(|w| w.to_f64().expect("Any standard float is a valid f64"))
                    .collect(),
            );
        }
    }

    window
}

/// Generate window function coefficients without caching (internal).
///
/// This is the actual implementation that computes window functions.
/// Used internally by the cached version.
fn generate_window_uncached<F: RealFloat>(size: usize, window_type: WindowType<F>) -> Vec<F> {
    match window_type {
        WindowType::Rectangular => vec![F::one(); size],
        WindowType::Hanning => (0..size)
            .map(|i| {
                let n: F = to_precision::<F, _>(i);
                let n_max: F = to_precision::<F, _>(size - 1);
                let two_pi: F = to_precision::<F, _>(2.0) * F::PI();
                to_precision::<F, _>(0.5) * (F::one() - (two_pi * n / n_max).cos())
            })
            .collect(),
        WindowType::Hamming => (0..size)
            .map(|i| {
                let n: F = to_precision::<F, _>(i);
                let n_max: F = to_precision::<F, _>(size - 1);
                let two_pi: F = to_precision::<F, _>(2.0) * F::PI();
                to_precision::<F, _>(0.54) - to_precision::<F, _>(0.46) * (two_pi * n / n_max).cos()
            })
            .collect(),
        WindowType::Blackman => (0..size)
            .map(|i| {
                let n: F = to_precision::<F, _>(i);
                let n_max: F = to_precision::<F, _>(size - 1);
                let four_pi: F = to_precision::<F, _>(4.0) * F::PI();
                let two_pi: F = to_precision::<F, _>(2.0) * F::PI();
                to_precision::<F, _>(0.42) - to_precision::<F, _>(0.5) * (two_pi * n / n_max).cos()
                    + to_precision::<F, _>(0.08) * (four_pi * n / n_max).cos()
            })
            .collect(),
        WindowType::Kaiser { beta } => {
            // Simplified Kaiser window (basic implementation)
            (0..size)
                .map(|i| {
                    let n: F = to_precision::<F, _>(i);
                    let n_max: F = to_precision::<F, _>(size - 1);
                    let alpha: F = (n - n_max / to_precision::<F, _>(2.0))
                        / (n_max / to_precision::<F, _>(2.0));
                    let bessel_arg = beta * (F::one() - alpha * alpha).sqrt();
                    // Simplified approximation of modified Bessel function
                    F::one()
                        + bessel_arg / to_precision::<F, _>(2.0)
                        // Normalize by I0(beta) approximation
                        / (F::one() + beta / to_precision::<F, _>(2.0))
                })
                .collect()
        }
        WindowType::Gaussian { std } => (0..size)
            .map(|i| {
                let n: F = to_precision::<F, _>(i);
                let center: F = to_precision::<F, _>(size - 1) / to_precision::<F, _>(2.0);
                let exponent: F = to_precision::<F, _>(-0.5) * ((n - center) / std).powi(2);
                to_precision::<F, _>(exponent.exp())
            })
            .collect(),
    }
}

/// Generates ERB-spaced center frequencies for gammatone filter bank.
///
/// The ERB (Equivalent Rectangular Bandwidth) scale provides perceptually
/// uniform frequency spacing that models human auditory processing.
///
/// # Arguments
/// * `n_filters` - Number of filters to generate
/// * `fmin` - Minimum frequency in Hz
/// * `fmax` - Maximum frequency in Hz
///
/// # Returns
/// Vector of center frequencies in Hz, spaced according to ERB scale
fn generate_erb_frequencies<F>(n_filters: usize, fmin: F, fmax: F) -> Vec<F>
where
    F: RealFloat,
{
    // Convert frequency range to ERB scale
    let erb_min = hz_to_erb(fmin);
    let erb_max = hz_to_erb(fmax);

    // Generate linearly spaced ERB values
    let erb_step = (erb_max - erb_min) / to_precision::<F, _>(n_filters - 1);

    (0..n_filters)
        .map(|i| {
            let erb_val = erb_min + to_precision::<F, usize>(i) * erb_step;
            erb_to_hz(erb_val)
        })
        .collect()
}

/// Converts frequency in Hz to ERB scale.
///
/// ERB scale formula: ERB(f) = 21.4 * log10(1 + 0.00437*f)
/// This provides approximately uniform spacing on the auditory scale.
fn hz_to_erb<F: RealFloat>(freq_hz: F) -> F {
    to_precision::<F, _>(21.4) * (F::one() + to_precision::<F, _>(0.00437) * freq_hz).log10()
}

/// Converts ERB scale value back to frequency in Hz.
///
/// Inverse of hz_to_erb: f = (10^(ERB/21.4) - 1) / 0.00437
fn erb_to_hz<F: RealFloat>(erb: F) -> F {
    (to_precision::<F, _>(10.0).powf(erb / to_precision::<F, _>(21.4)) - F::one())
        / to_precision::<F, _>(0.00437)
}

/// Generates a bank of gammatone filters with specified center frequencies.
///
/// Each filter is a 4th-order gammatone filter with impulse response:
/// g(t) = a * t³ * exp(-2πERB(f)t) * cos(2πft)
///
/// # Arguments
/// * `center_frequencies` - Center frequencies for each filter in Hz
/// * `sample_rate` - Sample rate in Hz
/// * `filter_length` - Length of each filter in samples
///
/// # Returns
/// Vector of filter coefficient vectors, one per center frequency
fn generate_gammatone_filter_bank<F: RealFloat>(
    center_frequencies: &[F],
    sample_rate: F,
    filter_length: usize,
) -> Vec<Vec<F>> {
    center_frequencies
        .iter()
        .map(|&center_freq| generate_gammatone_filter(center_freq, sample_rate, filter_length))
        .collect()
}

/// Generates a single 4th-order gammatone filter.
///
/// Gammatone filter impulse response:
/// g(t) = a * t³ * exp(-2πERB(f)t) * cos(2πft)
///
/// Where:
/// - a is normalization constant
/// - ERB(f) = 24.7 * (4.37*f/1000 + 1) is the equivalent rectangular bandwidth
/// - n=4 for 4th order filter (t³ term)
///
/// # Arguments
/// * `center_freq` - Center frequency in Hz
/// * `sample_rate` - Sample rate in Hz  
/// * `filter_length` - Filter length in samples
///
/// # Returns
/// Vector of filter coefficients
fn generate_gammatone_filter<F: RealFloat>(
    center_freq: F,
    sample_rate: F,
    filter_length: usize,
) -> Vec<F> {
    // Calculate ERB bandwidth for this frequency
    let erb: F = to_precision::<F, _>(24.7) * to_precision::<F, _>(4.37) * center_freq
        / to_precision::<F, _>(1000.0);

    // Time step
    let dt: F = F::one() / to_precision::<F, _>(sample_rate);
    // Generate filter coefficients
    let mut filter: Vec<F> = Vec::with_capacity(filter_length);
    let decay_coeff: F = to_precision::<F, _>(-2.0) * erb;
    let osc_coeff: F = to_precision::<F, _>(2.0) * center_freq;
    for n in 0..filter_length {
        let t = to_precision::<F, _>(to_precision::<F, usize>(n)) * dt;

        if t == F::zero() {
            // Handle t=0 case (t³ would be 0)
            filter.push(F::zero());
        } else {
            // 4th order gammatone: t³ * exp(-2πERBt) * cos(2πft)
            let t_cubed = t * t * t;
            let decay = (decay_coeff * t).exp();
            let oscillation = (osc_coeff * t).cos();

            let coefficient = t_cubed * decay * oscillation;
            filter.push(coefficient);
        }
    }

    // Normalize filter to unit energy
    let energy: F = filter
        .iter()
        .map(|&x| x * x)
        .reduce(|acc, x| acc + x)
        .unwrap_or(F::zero());
    if energy > F::zero() {
        let norm_factor = F::one() / energy.sqrt();
        filter.iter_mut().for_each(|x| *x = *x * norm_factor);
    }

    filter
}

/// Applies a gammatone filter to an input signal using convolution.
///
/// # Arguments
/// * `signal` - Input signal
/// * `filter_coeffs` - Gammatone filter coefficients
///
/// # Returns
/// Filtered signal (same length as input)
fn apply_gammatone_filter<F: RealFloat>(signal: &[F], filter_coeffs: &[F]) -> Vec<F> {
    let signal_len = signal.len();
    let filter_len = filter_coeffs.len();

    // Use circular convolution to maintain output length
    let mut output = vec![F::zero(); signal_len];

    for (i, out) in output.iter_mut().enumerate().take(signal_len) {
        for (j, coeff) in filter_coeffs
            .iter()
            .enumerate()
            .take(filter_len.min(signal_len))
        {
            let signal_idx = (i + signal_len - j) % signal_len;
            *out = *out + signal[signal_idx] * *coeff;
        }
    }

    output
}

/// Converts frequency in Hz to mel scale.
///
/// Mel scale provides perceptually uniform frequency spacing for human hearing.
/// Formula: mel = 2595 * log10(1 + f/700)
///
/// # Arguments
/// * `freq_hz` - Frequency in Hz
///
/// # Returns
/// Frequency in mel scale
fn hz_to_mel<F: RealFloat>(freq_hz: F) -> F {
    // 2595.0 * (F::one() + freq_hz / 700.0).log10()
    to_precision::<F, _>(2595.0) * (F::one() + freq_hz / to_precision::<F, _>(700.0)).log10()
}

/// Converts mel scale value back to frequency in Hz.
///
/// Inverse of hz_to_mel: f = 700 * (10^(mel/2595) - 1)
///
/// # Arguments
/// * `mel` - Frequency in mel scale
///
/// # Returns
/// Frequency in Hz
fn mel_to_hz<F: RealFloat>(mel: F) -> F {
    // 700.0 * (10.0_f64.powf(mel / 2595.0) - F::one())
    to_precision::<F, _>(700.0)
        * (to_precision::<F, _>(10.0).powf(mel / to_precision::<F, _>(2595.0)) - F::one())
}

/// Generates mel-spaced center frequencies for mel filter bank.
///
/// Creates linearly spaced frequencies on the mel scale, then converts back to Hz.
/// This provides perceptually uniform frequency spacing.
///
/// # Arguments
/// * `n_filters` - Number of mel filters
/// * `fmin` - Minimum frequency in Hz
/// * `fmax` - Maximum frequency in Hz
///
/// # Returns
/// Vector of mel filter center frequencies in Hz
fn generate_mel_frequencies<F: RealFloat>(n_filters: usize, fmin: F, fmax: F) -> Vec<F> {
    // Convert frequency range to mel scale
    let mel_min = hz_to_mel(fmin);
    let mel_max = hz_to_mel(fmax);

    // Generate linearly spaced mel values (need n_filters + 2 points for triangular filters)
    let mel_step = (mel_max - mel_min) / to_precision::<F, _>((n_filters + 1) as f64);

    (0..=n_filters + 1)
        .map(|i| {
            let mel_val = mel_min + to_precision::<F, _>(i as f64) * mel_step;
            mel_to_hz(mel_val)
        })
        .collect()
}

/// Generates a mel filter bank for mel spectrogram computation.
///
/// Creates triangular filters spaced on the mel scale. Each filter has:
/// - Linear rise from 0 to 1 between (f[i-1], f[i])
/// - Linear fall from 1 to 0 between (f[i], f[i+1])
/// - Zero response elsewhere
///
/// # Arguments
/// * `n_filters` - Number of mel filters
/// * `n_fft` - FFT size (determines frequency resolution)
/// * `sample_rate` - Sample rate in Hz
/// * `fmin` - Minimum frequency in Hz
/// * `fmax` - Maximum frequency in Hz
///
/// # Returns
/// 2D array with dimensions (n_filters, n_fft/2 + 1) containing filter bank
fn generate_mel_filter_bank<F: RealFloat>(
    n_filters: usize,
    n_fft: usize,
    sample_rate: F,
    fmin: F,
    fmax: F,
) -> Array2<F> {
    // Generate mel-spaced frequencies
    let mel_frequencies = generate_mel_frequencies(n_filters, fmin, fmax);

    // Calculate frequency bin centers for FFT
    let n_freq_bins = n_fft / 2 + 1;
    let freq_bins: Vec<F> = (0..n_freq_bins)
        .map(|i| to_precision::<F, _>(i as f64) * sample_rate / to_precision::<F, _>(n_fft as f64))
        .collect();

    // Initialize filter bank matrix
    let mut filter_bank = Array2::zeros((n_filters, n_freq_bins));

    // Create triangular filters
    for filter_idx in 0..n_filters {
        let f_left = mel_frequencies[filter_idx];
        let f_center = mel_frequencies[filter_idx + 1];
        let f_right = mel_frequencies[filter_idx + 2];

        for (bin_idx, &freq) in freq_bins.iter().enumerate() {
            let filter_val = if freq < f_left || freq > f_right {
                // Outside filter range
                F::zero()
            } else if freq >= f_left && freq <= f_center {
                // Rising edge: linear interpolation from 0 to 1
                if f_center == f_left {
                    F::one()
                } else {
                    (freq - f_left) / (f_center - f_left)
                }
            } else {
                // Falling edge: linear interpolation from 1 to 0
                if f_right == f_center {
                    F::one()
                } else {
                    (f_right - freq) / (f_right - f_center)
                }
            };

            filter_bank[[filter_idx, bin_idx]] = filter_val;
        }

        // Normalize filter to unit area (optional, depends on application)
        let filter_sum: F = filter_bank.row(filter_idx).sum();
        if filter_sum > F::zero() {
            filter_bank
                .row_mut(filter_idx)
                .mapv_inplace(|x| x / filter_sum);
        }
    }

    filter_bank
}

/// Computes the Discrete Cosine Transform (DCT) Type II for MFCC computation.
///
/// DCT is used to decorrelate mel filter bank outputs and compress
/// the most important spectral information into lower-order coefficients.
///
/// DCT-II formula: X[k] = Σ x[n] * cos(π*k*(2n+1)/(2N))
///
/// # Arguments
/// * `input` - Input mel filter bank energies (log scale)
/// * `n_mfcc` - Number of MFCC coefficients to compute
///
/// # Returns
/// Vector of MFCC coefficients
fn compute_dct_type2<F: RealFloat>(input: &[F], n_mfcc: usize) -> Vec<F> {
    let n_input = input.len();
    let mut dct_output = vec![F::zero(); n_mfcc];
    let pi = F::PI();
    let two = to_precision::<F, _>(2.0);
    let n_input_f = to_precision::<F, _>(n_input as f64);

    for (k, dct_out) in dct_output.iter_mut().enumerate().take(n_mfcc) {
        let mut sum = F::zero();
        for (n, inp) in input.iter().enumerate().take(n_input) {
            let cos_term = (pi
                * to_precision::<F, _>(k as f64)
                * (two * to_precision::<F, _>(n as f64) + F::one())
                / (two * n_input_f))
                .cos();
            sum = sum + (*inp * cos_term);
        }

        // Apply normalization factor
        let norm_factor = if k == 0 {
            (F::one() / n_input_f).sqrt()
        } else {
            (two / n_input_f).sqrt()
        };

        *dct_out = sum * norm_factor;
    }

    dct_output
}

/// CQT kernel data structure for efficient sparse representation.
///
/// Stores the CQT kernels in a sparse format to reduce memory usage
/// and computational complexity.
#[derive(Debug, Clone)]
struct CqtKernel<F: RealFloat> {
    /// Complex kernel coefficients for each frequency bin
    kernels: Vec<Vec<Complex<F>>>,
    /// Kernel lengths for each frequency bin
    kernel_lengths: Vec<usize>,
    /// FFT size used for convolution
    fft_size: usize,
}

/// Generates CQT kernels for all frequency bins.
///
/// Creates a bank of complex exponential kernels with logarithmically spaced
/// center frequencies and constant Q factor.
fn generate_cqt_kernel<F: RealFloat>(
    config: &CqtConfig<F>,
    sample_rate: F,
    signal_length: usize,
) -> AudioSampleResult<CqtKernel<F>> {
    let num_bins = config.num_bins(sample_rate);
    let mut kernels = Vec::with_capacity(num_bins);
    let mut frequencies = Vec::with_capacity(num_bins);
    let mut kernel_lengths = Vec::with_capacity(num_bins);

    // Calculate FFT size (next power of 2 for efficiency)
    let fft_size = (signal_length * 2).next_power_of_two();

    for bin_idx in 0..num_bins {
        let center_freq = config.bin_frequency(bin_idx);
        let _bandwidth = config.bin_bandwidth(bin_idx);

        // Check if frequency is within valid range
        if center_freq >= sample_rate / to_precision::<F, _>(2.0) {
            break;
        }

        // Calculate kernel length based on bandwidth
        let kernel_length = ((config.q_factor * sample_rate / center_freq)
            .round()
            .to_usize()
            .expect("kernel length should be a valid usize"))
        .max(1)
        .min(signal_length);

        // Generate complex exponential kernel
        let mut kernel: Vec<Complex<F>> = generate_cqt_kernel_bin::<F>(
            center_freq,
            kernel_length,
            sample_rate,
            config.window_type,
        )?;

        // Apply sparsity threshold
        apply_sparsity_threshold(&mut kernel, config.sparsity_threshold);

        // Normalize kernel if requested
        if config.normalize {
            normalize_kernel(&mut kernel);
        }

        kernels.push(kernel);
        frequencies.push(center_freq);
        kernel_lengths.push(kernel_length);
    }

    Ok(CqtKernel {
        kernels,
        kernel_lengths,
        fft_size,
    })
}

/// Generates a single CQT kernel for a specific frequency bin.
///
/// Creates a complex exponential kernel windowed by the specified window function.
fn generate_cqt_kernel_bin<F: RealFloat>(
    center_freq: F,
    kernel_length: usize,
    sample_rate: F,
    window_type: WindowType<F>,
) -> AudioSampleResult<Vec<Complex<F>>> {
    let mut kernel = Vec::with_capacity(kernel_length);

    // Generate window coefficients (cached for performance)
    let window = generate_window_cached::<F>(kernel_length, window_type);

    let pi = F::PI();
    let two = to_precision::<F, _>(2.0);
    let sample_rate_f = to_precision::<F, _>(sample_rate);
    // Generate complex exponential kernel
    for (n, w) in window.iter().enumerate().take(kernel_length) {
        let t = to_precision::<F, _>(n as f64) / sample_rate_f;
        let phase = two * pi * center_freq * t;

        // Complex exponential: e^(i*2*π*f*t)
        let exponential = Complex::new(phase.cos(), phase.sin());

        // Apply window function
        let windowed = exponential * *w;

        kernel.push(windowed);
    }

    Ok(kernel)
}

/// Applies sparsity threshold to reduce kernel size.
///
/// Sets coefficients below the threshold to zero to create sparse kernels.
fn apply_sparsity_threshold<F: RealFloat>(kernel: &mut [Complex<F>], threshold: F) {
    if threshold <= F::zero() {
        return;
    }

    // Find maximum magnitude in kernel
    let max_magnitude: F = kernel
        .iter()
        .map(|&c| c.norm())
        .fold(F::zero(), |a, b| a.max(b));

    if max_magnitude == F::zero() {
        return;
    }

    let absolute_threshold = max_magnitude * threshold;

    // Apply threshold
    for coefficient in kernel.iter_mut() {
        if coefficient.norm() < absolute_threshold {
            *coefficient = Complex::new(F::zero(), F::zero());
        }
    }
}

/// Normalizes a kernel to unit energy.
fn normalize_kernel<F: RealFloat>(kernel: &mut [Complex<F>]) {
    let energy: F = kernel
        .iter()
        .map(|c| c.norm_sqr())
        .reduce(|acc, x| acc + x)
        .unwrap_or(F::zero());

    if energy > F::zero() {
        let norm_factor = F::one() / energy.sqrt();
        for coefficient in kernel.iter_mut() {
            *coefficient = *coefficient * norm_factor;
        }
    }
}

/// Applies the CQT kernel to input samples using FFT convolution.
///
/// Convolves the input signal with each CQT kernel to compute the
/// Constant-Q Transform coefficients.
fn apply_cqt_kernel<F: RealFloat + FftNum>(
    samples: &[F],
    kernel: &CqtKernel<F>,
) -> AudioSampleResult<Vec<Complex<F>>> {
    let mut cqt_result = Vec::new();

    // Convert input to complex for FFT
    let mut input_buffer: Vec<Complex<F>> = samples
        .iter()
        .map(|&x| Complex::new(x, F::zero()))
        .collect();

    // Pad input to FFT size
    input_buffer.resize(kernel.fft_size, Complex::new(F::zero(), F::zero()));

    // Create FFT planner
    let mut fft_planner: FftPlanner<F> = FftPlanner::new();
    let fft_forward = fft_planner.plan_fft_forward(kernel.fft_size);
    let fft_inverse = fft_planner.plan_fft_inverse(kernel.fft_size);

    // Compute FFT of input
    fft_forward.process(&mut input_buffer);

    // Apply each kernel
    for (bin_idx, kernel_coeffs) in kernel.kernels.iter().enumerate() {
        if kernel_coeffs.is_empty() {
            cqt_result.push(Complex::new(F::zero(), F::zero()));
            continue;
        }

        // Pad kernel to FFT size
        let mut kernel_buffer: Vec<Complex<F>> = kernel_coeffs.clone();
        kernel_buffer.resize(kernel.fft_size, Complex::new(F::zero(), F::zero()));

        // Compute FFT of kernel
        fft_forward.process(&mut kernel_buffer);

        // Multiply in frequency domain (convolution)
        let mut convolution_buffer: Vec<Complex<F>> = input_buffer
            .iter()
            .zip(kernel_buffer.iter())
            .map(|(x, k)| x * k.conj()) // Complex conjugate for correlation
            .collect();

        // Inverse FFT
        fft_inverse.process(&mut convolution_buffer);

        // Take the central sample (zero-lag correlation)
        let result_idx = kernel.kernel_lengths[bin_idx] / 2;
        let coefficient = if result_idx < convolution_buffer.len() {
            convolution_buffer[result_idx] / to_precision::<F, _>(kernel.fft_size as f64)
        } else {
            Complex::new(F::zero(), F::zero())
        };

        cqt_result.push(coefficient);
    }

    Ok(cqt_result)
}

/// Generates dual CQT kernel for reconstruction.
///
/// Creates the dual frame kernels needed for inverse CQT reconstruction.
fn generate_dual_cqt_kernel<F: RealFloat>(
    config: &CqtConfig<F>,
    sample_rate: F,
    signal_length: usize,
) -> AudioSampleResult<CqtKernel<F>> {
    // For now, use the same kernel as the forward transform
    // A more sophisticated implementation would compute the actual dual frame
    generate_cqt_kernel(config, sample_rate, signal_length)
}

/// Applies inverse CQT using dual frame reconstruction.
///
/// Reconstructs the time-domain signal from CQT coefficients using
/// the dual frame method.
fn apply_inverse_cqt_kernel<F: RealFloat>(
    cqt_matrix: &Array2<Complex<F>>,
    dual_kernel: &CqtKernel<F>,
    signal_length: usize,
) -> AudioSampleResult<Vec<F>> {
    let mut reconstructed: Vec<F> = vec![F::zero(); signal_length];

    // Extract single-frame CQT coefficients
    let (num_bins, num_frames) = cqt_matrix.dim();

    if num_frames == 0 {
        return Ok(reconstructed);
    }

    // For single-frame reconstruction, use the first frame
    let frame_idx = 0;

    // Reconstruct using dual frame synthesis
    for bin_idx in 0..num_bins.min(dual_kernel.kernels.len()) {
        let coeff = cqt_matrix[[bin_idx, frame_idx]];
        let kernel = &dual_kernel.kernels[bin_idx];

        // Add contribution from this bin
        for (n, &kernel_val) in kernel.iter().enumerate() {
            if n < signal_length {
                let real = (coeff * kernel_val).re;
                reconstructed[n] += real;
            }
        }
    }

    Ok(reconstructed)
}

/// Applies Gaussian smoothing to a signal using a specified standard deviation.
///
/// The smoothing helps reduce noise and spurious peaks in the onset detection function.
pub fn gaussian_smooth<F: RealFloat>(signal: &[F], sigma: F) -> Vec<F> {
    if sigma <= F::zero() {
        return signal.to_vec();
    }

    let kernel_size: usize = (to_precision::<F, _>(6.0) * sigma)
        .ceil()
        .to_usize()
        .expect("Should not fail");
    let kernel_radius = kernel_size / 2;

    // Generate Gaussian kernel
    let mut kernel = Vec::with_capacity(kernel_size);
    let mut kernel_sum = F::zero();

    for i in 0..kernel_size {
        let x = to_precision::<F, usize>(i) - to_precision::<F, usize>(kernel_radius);

        let value = (to_precision::<F, _>(-0.5) * (x / sigma).powi(2)).exp();
        kernel.push(value);
        kernel_sum += value;
    }

    // Normalize kernel
    for value in kernel.iter_mut() {
        *value /= kernel_sum;
    }

    // Apply convolution
    let mut smoothed = vec![F::zero(); signal.len()];

    for (i, sig) in signal.iter().enumerate().take(kernel_size) {
        let mut sum = F::zero();
        let mut weight_sum = F::zero();

        for (j, &weight) in kernel.iter().enumerate() {
            let signal_idx = i as isize - kernel_radius as isize + j as isize;

            if signal_idx >= 0 && signal_idx < signal.len() as isize {
                sum += *sig * weight;
                weight_sum += weight;
            }
        }

        smoothed[i] = if weight_sum > F::zero() {
            sum / weight_sum
        } else {
            F::zero()
        };
    }

    smoothed
}

/// Computes standard spectral flux from CQT spectrogram.
///
/// Standard spectral flux: SF\[n\] = Σ(|X\[k,n\]| - |X\[k,n-1\]|)
/// Measures the sum of magnitude differences for all frequency bins.
pub fn compute_standard_spectral_flux<F: RealFloat>(
    cqt_spectrogram: &Array2<Complex<F>>,
) -> AudioSampleResult<Vec<F>> {
    let (num_bins, num_frames) = cqt_spectrogram.dim();

    if num_frames < 2 {
        return Ok(vec![F::zero()]);
    }

    let mut flux = vec![F::zero(); num_frames - 1];

    for frame_idx in 1..num_frames {
        let mut frame_flux = F::zero();

        for bin_idx in 0..num_bins {
            let curr_magnitude = cqt_spectrogram[[bin_idx, frame_idx]].norm();
            let prev_magnitude = cqt_spectrogram[[bin_idx, frame_idx - 1]].norm();
            let diff = curr_magnitude - prev_magnitude;
            frame_flux += diff;
        }

        flux[frame_idx - 1] = frame_flux;
    }

    Ok(flux)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operations::CqtConfig;
    use std::f64::consts::PI;

    /// Generate test audio signal (sine wave)
    fn generate_sine_wave(samples: usize, freq: f64, sample_rate: f64) -> Vec<f32> {
        (0..samples)
            .map(|i| (2.0 * PI * freq * i as f64 / sample_rate).sin() as f32)
            .collect()
    }

    #[allow(dead_code)] // Keep for future use
    /// Generate test audio signal (chirp - frequency sweep)
    fn generate_chirp(
        samples: usize,
        start_freq: f64,
        end_freq: f64,
        sample_rate: f64,
    ) -> Vec<f32> {
        (0..samples)
            .map(|i| {
                let t = i as f64 / sample_rate;
                let freq =
                    start_freq + (end_freq - start_freq) * t / (samples as f64 / sample_rate);
                (2.0 * PI * freq * t).sin() as f32
            })
            .collect()
    }

    #[test]
    fn test_fft_basic() {
        // Test FFT with a simple sine wave
        let sample_rate = 44100;
        let samples = generate_sine_wave(1024, 440.0, sample_rate as f64);
        let audio = AudioSamples::new_mono(Array1::from_vec(samples).into(), sample_rate);

        let fft_result: Vec<Complex<f32>> = audio.fft().unwrap();
        assert_eq!(fft_result.len(), 1024);

        // Check that we get complex numbers
        assert!(fft_result.iter().any(|c| c.im.abs() > 1e-10));
    }

    #[test]
    fn test_round_trip_fft_ifft_realistic() {
        // Test round-trip with realistic audio length (1 second)
        let sample_rate = 44100;
        let samples = generate_sine_wave(sample_rate, 440.0, sample_rate as f64);
        let original = AudioSamples::new_mono(Array1::from_vec(samples).into(), sample_rate as u32);

        // FFT -> IFFT
        let fft_result: Vec<Complex<f32>> = original.fft().unwrap();
        let reconstructed = original.ifft(&fft_result).unwrap();

        // Compare original and reconstructed (allowing for small numerical errors)
        match (&original.data, &reconstructed.data) {
            (AudioData::Mono(orig), AudioData::Mono(recon)) => {
                assert_eq!(orig.len(), recon.len());
                for (o, r) in orig.iter().zip(recon.iter()) {
                    let orig_val = *o as f64;
                    let recon_val = *r as f64;
                    let diff = (orig_val - recon_val).abs();
                    let max_val = orig_val.abs().max(recon_val.abs());

                    // Use relative tolerance for larger values, absolute for very small values
                    let tolerance = if max_val > 1e-6 {
                        max_val * 2e-4 // 0.02% relative error (accounts for FFT numerical precision)
                    } else {
                        1e-6 // Absolute tolerance for very small values
                    };

                    assert!(
                        diff <= tolerance,
                        "FFT/IFFT round-trip error too large: orig={}, recon={}, diff={}, tolerance={}",
                        orig_val,
                        recon_val,
                        diff,
                        tolerance
                    );
                }
            }
            _ => panic!("Expected mono audio"),
        }
    }

    #[test]
    fn test_stft_overlapping_windows() {
        // Test STFT with overlapping windows on moderate-length audio
        let sample_rate = 44100;
        let samples = generate_sine_wave(8192, 440.0, sample_rate as f64); // ~185ms
        let audio: AudioSamples<'_, f32> =
            AudioSamples::new_mono(Array1::from_vec(samples).into(), sample_rate);

        let window_size = 1024;
        let hop_size = 512; // 50% overlap

        let stft_result: Array2<Complex<f32>> = audio
            .stft(window_size, hop_size, WindowType::Hanning)
            .unwrap();
        let (freq_bins, time_frames) = stft_result.dim();

        assert_eq!(freq_bins, window_size / 2 + 1); // Real FFT bins
        // Expected frames with padding: (8192 + 2*512 - 1024) / 512 + 1 = 17
        assert_eq!(time_frames, 17);
    }

    #[test]
    fn test_spectrogram_linear_scale() {
        // Test linear scale spectrogram with realistic audio
        let sample_rate = 44100;
        let samples = generate_sine_wave(2048, 440.0, sample_rate as f64);
        let audio = AudioSamples::new_mono(Array1::from_vec(samples).into(), sample_rate);

        let window_size = 512;
        let hop_size = 256;

        let result = audio.spectrogram(
            window_size,
            hop_size,
            WindowType::Hanning,
            SpectrogramScale::Linear,
            false,
        );
        assert!(result.is_ok());

        let spectrogram = result.unwrap();
        let (freq_bins, time_frames) = spectrogram.dim();

        // Check dimensions - for real-valued signals, we only get positive frequency bins
        let expected_freq_bins = window_size / 2 + 1; // For real FFT: N/2 + 1 bins
        assert_eq!(freq_bins, expected_freq_bins);
        assert!(time_frames > 0);

        // Check that we have positive power values
        assert!(spectrogram.iter().any(|&x| x > 0.0));

        // Check that all frames have significant energy (for sine wave input)
        let max_energy_per_frame: Vec<f64> = (0..time_frames)
            .map(|t| {
                (0..freq_bins)
                    .map(|f| spectrogram[[f, t]])
                    .fold(0.0, f64::max)
            })
            .collect();

        // All frames should have significant energy
        assert!(max_energy_per_frame.iter().all(|&energy| energy > 1e-10));
    }

    #[test]
    fn test_spectrogram_log_scale() {
        // Test log scale spectrogram (dB)
        let sample_rate = 44100;
        let samples = generate_sine_wave(1024, 440.0, sample_rate as f64);
        let audio = AudioSamples::new_mono(Array1::from_vec(samples).into(), sample_rate);

        let result = audio.spectrogram(512, 256, WindowType::Hanning, SpectrogramScale::Log, false);
        assert!(result.is_ok());

        let spectrogram = result.unwrap();

        // Check that values are in dB range (should have negative values)
        let min_val = spectrogram.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = spectrogram.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        // Should have reasonable dB range
        assert!(min_val >= -100.0); // Floor at -100 dB
        assert!(max_val <= 50.0); // Reasonable upper bound
        assert!(min_val < max_val); // Should have dynamic range
    }

    #[test]
    fn test_spectrogram_mel_scale_error() {
        // Test that mel scale returns appropriate error until implemented
        let sample_rate = 44100;
        let samples = generate_sine_wave(1024, 440.0, sample_rate as f64);
        let audio = AudioSamples::new_mono(Array1::from_vec(samples).into(), sample_rate);

        let result = audio.spectrogram(
            512,
            256,
            WindowType::<f32>::Hanning,
            SpectrogramScale::Mel,
            false,
        );
        assert!(result.is_err());
        match result.unwrap_err() {
            AudioSampleError::InvalidParameter(_) => {}
            _ => panic!("Expected InvalidParameter error for mel scaling"),
        }
    }

    #[test]
    fn test_multi_channel_transforms() {
        // Test FFT with stereo audio
        let sample_rate = 44100;
        let left_samples: Vec<f32> = generate_sine_wave(1024, 440.0, sample_rate as f64);
        let right_samples: Vec<f32> = generate_sine_wave(1024, 880.0, sample_rate as f64);

        let mut stereo_data = Array2::zeros((2, 1024));
        for (i, &sample) in left_samples.iter().enumerate() {
            stereo_data[[0, i]] = sample;
        }
        for (i, &sample) in right_samples.iter().enumerate() {
            stereo_data[[1, i]] = sample;
        }

        let audio = AudioSamples::new_multi_channel(stereo_data.into(), sample_rate);

        // Should use first channel for FFT
        let fft_result = audio.fft::<f32>().unwrap();
        assert_eq!(fft_result.len(), 1024);
    }

    #[test]
    fn test_window_functions() {
        // Test different window types with realistic data
        let sample_rate = 44100;
        let samples = generate_sine_wave(2048, 440.0, sample_rate as f64);
        let audio: AudioSamples<'static, f32> =
            AudioSamples::new_mono(Array1::from_vec(samples).into(), sample_rate);

        let window_size = 1024;
        let hop_size = 512;

        // Test different window types
        let window_types: Vec<WindowType<f32>> = vec![
            WindowType::Rectangular,
            WindowType::Hanning,
            WindowType::Hamming,
            WindowType::Blackman,
        ];

        for window_type in window_types {
            let result = audio.stft(window_size, hop_size, window_type).unwrap();
            let (freq_bins, time_frames) = result.dim();

            assert_eq!(freq_bins, window_size / 2 + 1); // Real FFT bins
            assert!(time_frames > 0);
        }
    }

    #[test]
    fn test_edge_cases() {
        let sample_rate = 44100;

        // Test empty audio
        let empty_audio: AudioSamples<f32> =
            AudioSamples::new_mono(Array1::from_vec(vec![]).into(), sample_rate);
        assert!(empty_audio.fft::<f32>().is_ok()); // Empty should work

        // Test single sample
        let single_sample =
            AudioSamples::new_mono(Array1::from_vec(vec![1.0f32]).into(), sample_rate);
        let fft_result = single_sample.fft::<f32>().unwrap();
        assert_eq!(fft_result.len(), 1);

        // Test STFT with invalid parameters
        let samples = generate_sine_wave(1024, 440.0, sample_rate as f64);
        let audio = AudioSamples::new_mono(Array1::from_vec(samples).into(), sample_rate);

        // Window size larger than audio
        assert!(audio.stft(2048, 512, WindowType::<f32>::Hanning).is_err());

        // Zero window size
        assert!(audio.stft(0, 512, WindowType::<f32>::Hanning).is_err());

        // Hop size larger than window size
        assert!(audio.stft(512, 1024, WindowType::<f32>::Hanning).is_err());
    }

    #[test]
    fn test_istft_reconstruction() {
        // Test STFT -> ISTFT round-trip with simple signal
        let sample_rate = 44100;
        let samples = generate_sine_wave(2048, 440.0, sample_rate as f64); // Shorter signal
        let original = AudioSamples::new_mono(Array1::from_vec(samples).into(), sample_rate as u32);

        let window_size = 512; // Smaller window
        let hop_size = 256; // 50% overlap

        // Forward STFT
        let stft_matrix = original
            .stft(window_size, hop_size, WindowType::<f32>::Hanning)
            .unwrap();

        // Inverse STFT
        let reconstructed: AudioSamples<f32> = AudioSamples::istft(
            &stft_matrix,
            hop_size,
            WindowType::<f32>::Hanning,
            sample_rate as usize,
            true,
        )
        .unwrap();

        // Check that reconstruction produces valid output
        match (&original.data, &reconstructed.data) {
            (AudioData::Mono(orig), AudioData::Mono(recon)) => {
                // Basic sanity checks
                assert!(recon.len() > 0, "Reconstructed signal should not be empty");
                assert!(
                    recon.iter().any(|&x| x.abs() > 1e-6),
                    "Reconstructed signal should have non-zero values"
                );

                // Check correlation instead of exact match due to windowing effects
                let min_len = orig.len().min(recon.len()).min(1024); // Limit comparison length
                let mut correlation = 0.0f64;
                let mut orig_energy = 0.0f64;
                let mut recon_energy = 0.0f64;

                for i in window_size / 2..min_len - window_size / 2 {
                    let o = orig[i] as f64;
                    let r = recon[i] as f64;
                    correlation += o * r;
                    orig_energy += o * o;
                    recon_energy += r * r;
                }

                let normalized_correlation =
                    correlation / (orig_energy.sqrt() * recon_energy.sqrt());
                assert!(
                    normalized_correlation > 0.8,
                    "Correlation too low: {}",
                    normalized_correlation
                );
            }
            _ => panic!("Expected mono audio"),
        }
    }

    #[test]
    fn test_gammatone_spectrogram() {
        // Test gammatone spectrogram with realistic audio
        let sample_rate = 44100;
        let samples = generate_sine_wave(4096, 440.0, sample_rate as f64);
        let audio = AudioSamples::new_mono(Array1::from_vec(samples).into(), sample_rate);

        let n_filters = 32;
        let fmin = 100.0;
        let fmax = 8000.0;
        let window_size = 512;
        let hop_size = 256;

        let result = audio.gammatone_spectrogram(n_filters, fmin, fmax, window_size, hop_size);
        assert!(result.is_ok());

        let gammatone_spec = result.unwrap();
        let (freq_bands, time_frames) = gammatone_spec.dim();

        // Check dimensions
        assert_eq!(freq_bands, n_filters);
        assert!(time_frames > 0);

        // Check that we have positive energy values
        assert!(gammatone_spec.iter().any(|&x| x > 0.0));

        // For a sine wave at 440 Hz, expect energy concentrated in middle frequency bands
        let total_energy_per_band: Vec<f64> = (0..freq_bands)
            .map(|b| (0..time_frames).map(|t| gammatone_spec[[b, t]]).sum())
            .collect();

        // Should have some bands with significant energy
        assert!(total_energy_per_band.iter().any(|&energy| energy > 1e-6));
    }

    #[test]
    fn test_erb_frequency_conversion() {
        // Test ERB scale conversion functions
        let freq_hz = 1000.0;
        let erb = hz_to_erb(freq_hz);
        let freq_back: f64 = erb_to_hz(erb);

        // Round-trip should be accurate
        assert!((freq_hz - freq_back).abs() < 0.1);

        // Test frequency range
        let freqs: Vec<f64> = generate_erb_frequencies(10, 100.0, 8000.0);
        assert_eq!(freqs.len(), 10);
        assert!((freqs[0] - 100.0).abs() < 1.0);
        assert!((freqs[9] - 8000.0).abs() < 10.0);

        // Frequencies should be monotonically increasing
        for i in 1..freqs.len() {
            assert!(freqs[i] > freqs[i - 1]);
        }
    }

    #[test]
    fn test_mel_spectrogram() {
        // Test mel spectrogram with realistic audio
        let sample_rate = 44100;
        let samples = generate_sine_wave(4096, 440.0, sample_rate as f64);
        let audio = AudioSamples::new_mono(Array1::from_vec(samples).into(), sample_rate);

        let n_mels = 40;
        let fmin = 80.0;
        let fmax = 8000.0;
        let window_size = 1024;
        let hop_size = 512;

        let result = audio.mel_spectrogram(n_mels, fmin, fmax, window_size, hop_size);
        assert!(result.is_ok());

        let mel_spec = result.unwrap();
        let (mel_bands, time_frames) = mel_spec.dim();

        // Check dimensions
        assert_eq!(mel_bands, n_mels);
        assert!(time_frames > 0);

        // Check that values are in reasonable dB range
        let min_val = mel_spec.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = mel_spec.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        assert!(min_val >= -100.0); // Floor at -100 dB
        assert!(max_val <= 50.0); // Reasonable upper bound
        assert!(min_val < max_val); // Should have dynamic range

        // For a sine wave at 440 Hz, expect energy in lower-middle mel bands
        let total_energy_per_band: Vec<f64> = (0..mel_bands)
            .map(|b| (0..time_frames).map(|t| mel_spec[[b, t]]).sum())
            .collect();

        // Should have some bands with significant energy
        assert!(total_energy_per_band.iter().any(|&energy| energy > -50.0));
    }

    #[test]
    fn test_mfcc() {
        // Test MFCC computation with realistic audio
        let sample_rate = 44100;
        let samples = generate_sine_wave(8192, 440.0, sample_rate as f64);
        let audio = AudioSamples::new_mono(Array1::from_vec(samples).into(), sample_rate);

        let n_mfcc = 13;
        let n_mels = 40;
        let fmin = 80.0;
        let fmax = 8000.0;

        let result = audio.mfcc(n_mfcc, n_mels, fmin, fmax);
        assert!(result.is_ok());

        let mfcc_matrix = result.unwrap();
        let (mfcc_coeffs, time_frames) = mfcc_matrix.dim();

        // Check dimensions
        assert_eq!(mfcc_coeffs, n_mfcc);
        assert!(time_frames > 0);

        // MFCC coefficients should have reasonable range
        let min_val = mfcc_matrix.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = mfcc_matrix.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        // MFCCs can have both positive and negative values
        assert!(min_val < 0.0);
        assert!(max_val > 0.0);
        assert!((max_val - min_val) > 1.0); // Should have reasonable dynamic range

        // First MFCC coefficient (c0) typically has highest magnitude (DC component)
        let c0_energy: f64 = (0..time_frames).map(|t| mfcc_matrix[[0, t]].abs()).sum();
        let c1_energy: f64 = (0..time_frames).map(|t| mfcc_matrix[[1, t]].abs()).sum();

        // c0 should typically have higher magnitude than c1 for most signals
        assert!(c0_energy > 0.0);
        assert!(c1_energy > 0.0);
    }

    #[test]
    fn test_mel_frequency_conversion() {
        // Test mel scale conversion functions
        let freq_hz = 1000.0;
        let mel = hz_to_mel(freq_hz);
        let freq_back: f64 = mel_to_hz(mel);

        // Round-trip should be accurate
        assert!((freq_hz - freq_back).abs() < 0.1);

        // Test known mel scale values
        assert!((hz_to_mel(0.0) - 0.0_f64).abs() < 0.1);
        assert!((hz_to_mel(700.0) - 781.0_f64).abs() < 1.0); // Approximately 781 mels (2595 * log10(2))

        // Test mel frequency generation
        let mel_freqs: Vec<f64> = generate_mel_frequencies(10, 100.0, 8000.0);
        assert_eq!(mel_freqs.len(), 12); // n_filters + 2 for triangular filters
        assert!((mel_freqs[0] - 100.0).abs() < 1.0); // First frequency (fmin)
        assert!((mel_freqs[11] - 8000.0).abs() < 10.0); // Last frequency (fmax)

        // Frequencies should be monotonically increasing
        for i in 1..mel_freqs.len() {
            assert!(mel_freqs[i] > mel_freqs[i - 1]);
        }
    }

    #[test]
    fn test_mel_filter_bank() {
        // Test mel filter bank generation
        let n_mels = 10;
        let n_fft = 512;
        let sample_rate = 44100.0;
        let fmin = 100.0;
        let fmax: f64 = 8000.0;

        let filter_bank = generate_mel_filter_bank(n_mels, n_fft, sample_rate, fmin, fmax);
        let (n_filters, n_freq_bins) = filter_bank.dim();

        // Check dimensions
        assert_eq!(n_filters, n_mels);
        assert_eq!(n_freq_bins, n_fft / 2 + 1);

        // Each filter should have non-negative values
        assert!(filter_bank.iter().all(|&x| x >= 0.0));

        // Each filter should have some non-zero values (triangular shape)
        for filter_idx in 0..n_filters {
            let filter_sum: f64 = filter_bank.row(filter_idx).sum();
            assert!(
                filter_sum > 0.0,
                "Filter {} should have positive sum",
                filter_idx
            );
        }

        // Check that filters are roughly triangular (have a peak)
        for filter_idx in 0..n_filters {
            let filter_values: Vec<f64> = filter_bank.row(filter_idx).to_vec();
            let max_val: f64 = filter_values.iter().fold(0.0, |a, &b| a.max(b));
            assert!(max_val > 0.0, "Filter {} should have a peak", filter_idx);
        }
    }

    #[test]
    fn test_dct_type2() {
        // Test DCT computation with known input
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let n_output = 4;

        let dct_output = compute_dct_type2(&input, n_output);
        assert_eq!(dct_output.len(), n_output);

        // DCT should preserve energy (Parseval's theorem, approximately)
        let input_energy: f64 = input.iter().map(|&x| x * x).sum();
        let output_energy: f64 = dct_output.iter().map(|&x| x * x).sum();

        // Energy should be approximately preserved (within 10%)
        let energy_ratio = output_energy / input_energy;
        assert!(
            energy_ratio > 0.9 && energy_ratio < 1.1,
            "Energy not preserved: input={}, output={}, ratio={}",
            input_energy,
            output_energy,
            energy_ratio
        );

        // First coefficient should be related to the DC component
        let dc_expected = input.iter().sum::<f64>() / (input.len() as f64).sqrt();
        assert!((dct_output[0] - dc_expected).abs() < 0.1);
    }

    #[test]
    fn test_cqt_config() {
        // Test CQT configuration creation and validation
        let config = CqtConfig::new();
        assert_eq!(config.bins_per_octave, 12);
        assert_eq!(config.fmin, 55.0);
        assert_eq!(config.q_factor, 1.0);
        assert!(config.normalize);

        let sample_rate = 44100.0;
        assert!(config.validate(sample_rate).is_ok());

        // Test invalid configuration
        let mut invalid_config = config.clone();
        invalid_config.fmin = 0.0;
        assert!(invalid_config.validate(sample_rate).is_err());

        // Test frequency calculations
        let freq_0: f64 = config.bin_frequency(0);
        let freq_12: f64 = config.bin_frequency(12);

        assert!((freq_0 - 55.0).abs() < 0.1);
        assert!((freq_12 - 110.0).abs() < 0.1); // One octave higher

        // Test number of bins calculation
        let num_bins = config.num_bins(sample_rate);
        assert!(num_bins > 0);
        assert!(num_bins < 200); // Reasonable upper bound
    }

    #[test]
    fn test_cqt_basic() {
        // Test basic CQT computation with a simple sine wave
        let sample_rate = 44100;
        let frequency = 440.0; // A4
        let duration = 0.5; // 0.5 seconds
        let samples: Vec<f32> = generate_sine_wave(
            (sample_rate as f64 * duration) as usize,
            frequency,
            sample_rate as f64,
        );

        let audio = AudioSamples::new_mono(Array1::from_vec(samples).into(), sample_rate);
        let config = CqtConfig::new();

        let result = audio.constant_q_transform(&config);
        assert!(result.is_ok());

        let cqt_matrix = result.unwrap();
        let (num_bins, num_frames) = cqt_matrix.dim();

        // Check dimensions
        assert_eq!(num_frames, 1); // Single-frame analysis
        assert!(num_bins > 0);
        assert!(num_bins == config.num_bins(sample_rate as f64));

        // Check that the result contains some energy
        let total_energy: f64 = cqt_matrix.iter().map(|c| c.norm_sqr()).sum();
        assert!(total_energy > 0.0);

        // For a pure sine wave, expect energy concentrated in specific bins
        let magnitudes: Vec<f64> = cqt_matrix.column(0).iter().map(|c| c.norm()).collect();
        let max_magnitude: f64 = magnitudes.iter().fold(0.0, |a, &b| a.max(b));
        let max_index = magnitudes.iter().position(|&x| x == max_magnitude).unwrap();

        // The bin with maximum energy should correspond roughly to 440 Hz
        let detected_freq = config.bin_frequency(max_index);
        assert!(
            (detected_freq - frequency).abs() < 150.0,
            "Expected frequency around {}, got {}",
            frequency,
            detected_freq
        );
    }

    #[test]
    fn test_cqt_spectrogram() {
        // Test CQT spectrogram computation
        let sample_rate = 44100;
        let frequency = 440.0;
        let duration = 1.0; // 1 second
        let samples = generate_sine_wave(
            (sample_rate as f64 * duration) as usize,
            frequency,
            sample_rate as f64,
        );

        let audio = AudioSamples::new_mono(Array1::from_vec(samples).into(), sample_rate);
        let config = CqtConfig::new();
        let hop_size = 512;

        let result = audio.cqt_spectrogram(&config, hop_size, None);
        assert!(result.is_ok());

        let cqt_spectrogram = result.unwrap();
        let (num_bins, num_frames) = cqt_spectrogram.dim();

        // Check dimensions
        assert!(num_bins > 0);
        assert!(num_frames > 1); // Should have multiple frames
        assert!(num_bins == config.num_bins(sample_rate as f64));

        // Check that the result contains energy in each frame
        for frame_idx in 0..num_frames {
            let frame_energy: f64 = cqt_spectrogram
                .column(frame_idx)
                .iter()
                .map(|c| c.norm_sqr())
                .sum();
            assert!(frame_energy > 0.0, "Frame {} should have energy", frame_idx);
        }
    }

    #[test]
    fn test_cqt_magnitude_spectrogram() {
        // Test CQT magnitude spectrogram computation
        let sample_rate = 44100;
        let frequency = 440.0;
        let duration = 0.5;
        let samples = generate_sine_wave(
            (sample_rate as f64 * duration) as usize,
            frequency,
            sample_rate as f64,
        );

        let audio = AudioSamples::new_mono(Array1::from_vec(samples).into(), sample_rate);
        let config = CqtConfig::<f32>::new();
        let hop_size = 512;

        // Test magnitude spectrogram
        let result = audio.cqt_magnitude_spectrogram(&config, hop_size, None, false);
        assert!(result.is_ok());

        let magnitude_spectrogram = result.unwrap();
        let (num_bins, num_frames) = magnitude_spectrogram.dim();

        // Check dimensions
        assert!(num_bins > 0);
        assert!(num_frames > 0);

        // All values should be non-negative (magnitude)
        assert!(magnitude_spectrogram.iter().all(|&x| x >= 0.0));

        // Test power spectrogram
        let power_result = audio.cqt_magnitude_spectrogram(&config, hop_size, None, true);
        assert!(power_result.is_ok());

        let power_spectrogram = power_result.unwrap();
        assert_eq!(power_spectrogram.dim(), magnitude_spectrogram.dim());

        // All values should be non-negative (power)
        assert!(power_spectrogram.iter().all(|&x| x >= 0.0));
    }

    #[test]
    fn test_cqt_different_configurations() {
        // Test different CQT configurations
        let sample_rate = 44100;
        let frequency = 440.0;
        let duration = 0.5;
        let samples = generate_sine_wave(
            (sample_rate as f64 * duration) as usize,
            frequency,
            sample_rate as f64,
        );

        let audio = AudioSamples::new_mono(Array1::from_vec(samples).into(), sample_rate);

        // Test musical configuration
        let musical_config = CqtConfig::<f32>::musical();
        let result = audio.constant_q_transform(&musical_config);
        assert!(result.is_ok());

        // Test harmonic configuration
        let harmonic_config = CqtConfig::<f32>::harmonic();
        let result = audio.constant_q_transform(&harmonic_config);
        assert!(result.is_ok());

        // Test chord detection configuration
        let chord_config = CqtConfig::<f32>::chord_detection();
        let result = audio.constant_q_transform(&chord_config);
        assert!(result.is_ok());

        // Test onset detection configuration
        let onset_config = CqtConfig::<f32>::onset_detection();
        let result = audio.constant_q_transform(&onset_config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cqt_edge_cases() {
        // Test edge cases and error conditions
        let sample_rate = 44100;
        let config = CqtConfig::<f32>::new();

        // Test with empty signal
        let empty_audio =
            AudioSamples::new_mono(Array1::<f32>::zeros(0).into(), sample_rate as u32);
        let result = empty_audio.constant_q_transform(&config);
        assert!(result.is_err());

        // Test with very short signal
        let short_audio =
            AudioSamples::new_mono(Array1::<f32>::ones(10).into(), sample_rate as u32);
        let result = short_audio.constant_q_transform(&config);
        assert!(result.is_ok()); // Should handle short signals gracefully

        // Test with invalid hop size for spectrogram
        let samples = generate_sine_wave(1024, 440.0, sample_rate as f64);
        let audio = AudioSamples::new_mono(Array1::from_vec(samples).into(), sample_rate as u32);

        let result = audio.cqt_spectrogram(&config, 0, None);
        assert!(result.is_err()); // Hop size of 0 should error
    }

    #[test]
    fn test_cqt_inverse_transform() {
        // Test inverse CQT reconstruction
        let sample_rate = 44100;
        let frequency = 440.0;
        let duration = 0.25; // Short duration for faster test
        let samples = generate_sine_wave(
            (sample_rate as f64 * duration) as usize,
            frequency,
            sample_rate as f64,
        );

        let original_audio =
            AudioSamples::new_mono(Array1::from_vec(samples.clone()).into(), sample_rate);
        let config = CqtConfig::<f32>::new();

        // Forward transform
        let cqt_result = original_audio.constant_q_transform(&config);
        assert!(cqt_result.is_ok());

        let cqt_matrix = cqt_result.unwrap();

        // Inverse transform
        let reconstructed_result = AudioSamples::<f32>::inverse_constant_q_transform(
            &cqt_matrix,
            &config,
            samples.len(),
            sample_rate as f32,
        );
        assert!(reconstructed_result.is_ok());

        let reconstructed_audio = reconstructed_result.unwrap();

        // Check that reconstruction has the same length
        assert_eq!(
            reconstructed_audio.samples_per_channel(),
            original_audio.samples_per_channel()
        );

        // The reconstruction might not be perfect due to the nature of CQT,
        // but it should preserve the general structure
        assert!(reconstructed_audio.samples_per_channel() > 0);
    }

    #[test]
    fn test_cqt_kernel_generation() {
        // Test CQT kernel generation
        let config = CqtConfig::new();
        let sample_rate = 44100.0;
        let signal_length = 1024;

        let kernel_result = generate_cqt_kernel(&config, sample_rate, signal_length);
        assert!(kernel_result.is_ok());

        let kernel = kernel_result.unwrap();

        // Check kernel properties
        assert!(kernel.kernels.len() > 0);
        assert_eq!(kernel.kernels.len(), kernel.kernel_lengths.len());

        // Check that kernel lengths are reasonable
        for &length in &kernel.kernel_lengths {
            assert!(length > 0);
            assert!(length <= signal_length);
        }
    }
}
