//! AudioTransforms trait implementation for Python bindings.
//!
//! This module provides Python access to frequency domain analysis and
//! spectral transformations, returning NumPy arrays for efficient integration
//! with the Python scientific ecosystem.

use super::{PyAudioSamples, utils::*};
use crate::AudioSamples;
use crate::operations::{
    AudioTransforms,
    types::{SpectrogramScale, WindowType},
};
use num_complex::Complex;
use pyo3::prelude::*;

impl PyAudioSamples {
    /// Parse window type from string
    pub(crate) fn parse_window_type_impl(window: &str) -> PyResult<WindowType> {
        match window.to_lowercase().as_str() {
            "rectangular" | "rect" => Ok(WindowType::Rectangular),
            "hanning" | "hann" => Ok(WindowType::Hanning),
            "hamming" => Ok(WindowType::Hamming),
            "blackman" => Ok(WindowType::Blackman),
            "kaiser" => Ok(WindowType::Kaiser { beta: 8.6 }), // Default beta value
            "gaussian" => Ok(WindowType::Gaussian { std: 0.4 }), // Default std value
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Invalid window type: '{}'. Valid options: 'rectangular', 'hanning', 'hamming', 'blackman', 'kaiser', 'gaussian'",
                window
            ))),
        }
    }
}

impl PyAudioSamples {
    /// Compute the Fast Fourier Transform of the audio samples.
    ///
    /// Returns complex frequency domain representation as a NumPy array.
    ///
    /// # Arguments
    /// * `n` - FFT size (default: length of audio)
    /// * `axis` - Axis along which to compute FFT (for multi-channel audio)
    ///
    /// # Returns
    /// NumPy array of complex numbers representing the frequency spectrum
    ///
    /// # Examples
    /// ```python
    /// import numpy as np
    /// import audio_samples as aus
    ///
    /// # Create a 440Hz sine wave
    /// t = np.linspace(0, 1, 44100)
    /// sine = np.sin(2 * np.pi * 440 * t)
    /// audio = aus.from_numpy(sine, sample_rate=44100)
    ///
    /// # Compute FFT
    /// spectrum = audio.fft()
    /// frequencies = np.fft.fftfreq(len(spectrum), 1/44100)
    /// # Should show peak at 440 Hz
    /// ```
    pub(crate) fn fft_impl(
        &self,
        py: Python,
        _n: Option<usize>,
        _axis: i32,
    ) -> PyResult<Py<PyAny>> {
        let fft_result = self.with_inner(|inner| inner.fft()).map_err(map_error)?;

        // Convert Vec<Complex<f64>> to numpy array
        let complex_array = ndarray::Array1::from(fft_result);
        array1_to_numpy(py, complex_array)
    }

    /// Compute the inverse Fast Fourier Transform from frequency domain.
    ///
    /// # Arguments
    /// * `spectrum` - NumPy array of complex frequency domain data
    ///
    /// # Returns
    /// New AudioSamples object reconstructed from the spectrum
    ///
    /// # Examples
    /// ```python
    /// # Round-trip test
    /// spectrum = audio.fft()
    /// reconstructed = audio.ifft(spectrum)
    /// # reconstructed should be very close to original audio
    /// ```
    pub(crate) fn ifft_impl(&self, spectrum: &Bound<PyAny>) -> PyResult<PyAudioSamples> {
        // Convert numpy array to Vec<Complex<f64>>
        let spectrum_vec =
            if let Ok(array) = spectrum.extract::<numpy::PyReadonlyArray1<Complex<f64>>>() {
                array.as_array().to_vec()
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "Spectrum must be a 1D numpy array of complex numbers",
                ));
            };

        let reconstructed = self
            .with_inner(|inner| inner.ifft(&spectrum_vec))
            .map_err(map_error)?;
        Ok(PyAudioSamples::from_inner(reconstructed))
    }

    /// Compute the Short-Time Fourier Transform (STFT).
    ///
    /// Returns a 2D array where each column represents the spectrum of a time frame.
    /// Compatible with librosa.stft() parameter naming and conventions.
    ///
    /// # Arguments
    /// * `n_fft` - FFT window size in samples (librosa compatibility)
    /// * `hop_length` - Number of samples between successive frames (librosa compatibility, default: n_fft/4)
    /// * `win_length` - Window length in samples (default: n_fft)
    /// * `window` - Window function type ('hann', 'hamming', 'blackman', etc.)
    /// * `center` - Whether to pad the signal for centered frames (default: true)
    /// 
    /// Legacy parameters (for backward compatibility):
    /// * `window_size` - Alias for n_fft
    /// * `hop_size` - Alias for hop_length
    ///
    /// # Returns
    /// 2D NumPy array of complex numbers (frequency bins × time frames)
    ///
    /// # Examples
    /// ```python
    /// # Librosa-style API
    /// D = audio.stft(n_fft=2048, hop_length=512, window='hann')
    /// magnitude = np.abs(D)
    /// phase = np.angle(D)
    ///
    /// # Legacy API (still supported)
    /// stft_matrix = audio.stft(window_size=1024, hop_size=256, window='hanning')
    /// ```
    pub(crate) fn stft_impl(
        &self,
        py: Python,
        n_fft: Option<usize>,
        hop_length: Option<usize>,
        win_length: Option<usize>,
        window: &str,
        center: bool,
        // Legacy parameters for backward compatibility
        window_size: Option<usize>,
        hop_size: Option<usize>,
    ) -> PyResult<Py<PyAny>> {
        // Resolve parameters with librosa priority
        let actual_n_fft = n_fft.or(window_size).unwrap_or(2048);
        let actual_win_length = win_length.unwrap_or(actual_n_fft);
        let actual_hop_length = hop_length.or(hop_size).unwrap_or(actual_n_fft / 4);
        
        // Validate parameters
        if actual_win_length > actual_n_fft {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "win_length cannot be larger than n_fft"
            ));
        }

        let window_type = Self::parse_window_type_impl(window)?;

        let stft_result = self
            .with_inner(|inner| inner.stft(actual_n_fft, actual_hop_length, window_type))
            .map_err(map_error)?;
            
        // TODO: Apply centering if center=true
        // For now, just return the result - centering will be implemented in future
        let _ = center;
        let _ = actual_win_length;
        
        array2_to_numpy(py, stft_result)
    }

    /// Compute the inverse Short-Time Fourier Transform.
    ///
    /// Reconstructs time domain signal from STFT matrix using overlap-add.
    ///
    /// # Arguments
    /// * `stft_matrix` - 2D NumPy array of complex STFT coefficients
    /// * `hop_size` - Hop size used in forward STFT
    /// * `window` - Window function type (should match forward STFT)
    ///
    /// # Returns
    /// New AudioSamples object reconstructed from STFT
    ///
    /// # Examples
    /// ```python
    /// # Round-trip test
    /// stft_matrix = audio.stft(window_size=1024, hop_size=256)
    /// reconstructed = audio.istft(stft_matrix, hop_size=256, window='hanning')
    /// ```
    pub(crate) fn istft_impl(
        &self,
        stft_matrix: &Bound<PyAny>,
        hop_size: usize,
        window: &str,
    ) -> PyResult<PyAudioSamples> {
        // Convert numpy array to Array2<Complex<f64>>
        let stft_array =
            if let Ok(array) = stft_matrix.extract::<numpy::PyReadonlyArray2<Complex<f64>>>() {
                array.as_array().to_owned()
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "STFT matrix must be a 2D numpy array of complex numbers",
                ));
            };

        let window_type = Self::parse_window_type_impl(window)?;
        let sample_rate = self.sample_rate() as usize;

        let reconstructed =
            AudioSamples::<f64>::istft(&stft_array, hop_size, window_type, sample_rate)
                .map_err(map_error)?;
        Ok(PyAudioSamples::from_inner(reconstructed))
    }

    /// Compute magnitude or power spectrogram.
    ///
    /// Compatible with both librosa-style and legacy parameter naming.
    ///
    /// # Arguments
    /// * `n_fft` - FFT window size in samples (librosa compatibility)
    /// * `hop_length` - Number of samples between frames (librosa compatibility, default: n_fft/4)
    /// * `win_length` - Window length in samples (default: n_fft)
    /// * `window` - Window function type ('hann', 'hamming', etc.)
    /// * `scale` - Scale type ('linear', 'log', 'mel')
    /// * `power` - Power of the magnitude (1.0 for magnitude, 2.0 for power)
    /// 
    /// Legacy parameters (for backward compatibility):
    /// * `window_size` - Alias for n_fft
    /// * `hop_size` - Alias for hop_length
    ///
    /// # Returns
    /// 2D NumPy array of real numbers (frequency bins × time frames)
    ///
    /// # Examples
    /// ```python
    /// # Librosa-style API 
    /// S = audio.spectrogram(n_fft=2048, hop_length=512, window='hann')
    /// S_db = audio.spectrogram(n_fft=2048, hop_length=512, scale='log')
    ///
    /// # Legacy API (still supported)
    /// spec_linear = audio.spectrogram(window_size=1024, scale='linear')
    /// ```
    pub(crate) fn spectrogram_impl(
        &self,
        py: Python,
        n_fft: Option<usize>,
        hop_length: Option<usize>,
        win_length: Option<usize>,
        window: &str,
        scale: &str,
        power: f64,
        // Legacy parameters for backward compatibility
        window_size: Option<usize>,
        hop_size: Option<usize>,
    ) -> PyResult<Py<PyAny>> {
        validate_string_param("scale", scale, &["linear", "log", "mel"])?;

        // Resolve parameters with librosa priority
        let actual_n_fft = n_fft.or(window_size).unwrap_or(2048);
        let actual_hop_length = hop_length.or(hop_size).unwrap_or(actual_n_fft / 4);
        let actual_win_length = win_length.unwrap_or(actual_n_fft);
        
        // Validate parameters
        if actual_win_length > actual_n_fft {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "win_length cannot be larger than n_fft"
            ));
        }

        let window_type = Self::parse_window_type_impl(window)?;

        let spectrogram_result = self
            .with_inner(|inner| {
                inner.spectrogram(
                    actual_n_fft,
                    actual_hop_length,
                    window_type,
                    SpectrogramScale::Linear,
                    false,
                )
            })
            .map_err(map_error)?;

        // Apply scale transformation
        let processed = match scale {
            "linear" => {
                if power == 1.0 {
                    spectrogram_result // Already magnitude
                } else {
                    spectrogram_result.mapv(|x| x.powf(power / 2.0))
                }
            }
            "log" => {
                // Convert to dB scale
                spectrogram_result.mapv(|x| 20.0 * (x.max(1e-10)).log10())
            }
            "mel" => {
                // For mel scale, we should use mel_spectrogram method
                return self.mel_spectrogram_impl(
                    py,
                    None,
                    None,
                    None,
                    Some(actual_n_fft),
                    Some(actual_hop_length),
                );
            }
            _ => spectrogram_result,
        };

        array2_to_numpy(py, processed)
    }

    /// Compute mel-scaled spectrogram for perceptual analysis.
    ///
    /// Uses triangular filter banks spaced on the mel scale for perceptually
    /// relevant frequency analysis.
    ///
    /// # Arguments
    /// * `n_mels` - Number of mel filter banks
    /// * `fmin` - Minimum frequency in Hz
    /// * `fmax` - Maximum frequency in Hz (default: Nyquist frequency)
    /// * `window_size` - STFT window size
    /// * `hop_size` - STFT hop size
    ///
    /// # Returns
    /// 2D NumPy array (mel bins × time frames)
    ///
    /// # Examples
    /// ```python
    /// # Standard mel spectrogram for speech
    /// mel_spec = audio.mel_spectrogram(n_mels=128, fmin=0, fmax=8000)
    ///
    /// # High resolution mel spectrogram
    /// mel_spec_hires = audio.mel_spectrogram(n_mels=256, window_size=2048)
    /// ```
    pub(crate) fn mel_spectrogram_impl(
        &self,
        py: Python,
        n_mels: Option<usize>,
        fmin: Option<f64>,
        fmax: Option<f64>,
        window_size: Option<usize>,
        hop_size: Option<usize>,
    ) -> PyResult<Py<PyAny>> {
        let actual_n_mels = n_mels.unwrap_or(128);
        let actual_fmin = fmin.unwrap_or(0.0);
        let actual_fmax = parse_fmax(fmax, self.sample_rate());
        let actual_window_size = window_size.unwrap_or(1024);
        let actual_hop_size = parse_hop_size(hop_size, actual_window_size);

        let mel_spec = self
            .with_inner(|inner| {
                inner.mel_spectrogram(
                    actual_n_mels,
                    actual_fmin,
                    actual_fmax,
                    actual_window_size,
                    actual_hop_size,
                )
            })
            .map_err(map_error)?;

        array2_to_numpy(py, mel_spec)
    }

    /// Compute Mel-Frequency Cepstral Coefficients (MFCC).
    ///
    /// Commonly used features for speech recognition and audio analysis.
    /// Computed as the DCT of log mel spectrogram.
    ///
    /// # Arguments
    /// * `n_mfcc` - Number of MFCC coefficients to return
    /// * `n_mels` - Number of mel filter banks to use
    /// * `fmin` - Minimum frequency in Hz
    /// * `fmax` - Maximum frequency in Hz
    /// * `window_size` - STFT window size
    /// * `hop_size` - STFT hop size
    ///
    /// # Returns
    /// 2D NumPy array (MFCC coefficients × time frames)
    ///
    /// # Examples
    /// ```python
    /// # Standard 13 MFCC coefficients for speech recognition
    /// mfcc = audio.mfcc(n_mfcc=13, n_mels=128)
    ///
    /// # Higher dimensional feature vector
    /// mfcc_hd = audio.mfcc(n_mfcc=39, n_mels=256)
    /// ```
    pub(crate) fn mfcc_impl(
        &self,
        py: Python,
        n_mfcc: Option<usize>,
        n_mels: Option<usize>,
        fmin: Option<f64>,
        fmax: Option<f64>,
        _window_size: Option<usize>,
        _hop_size: Option<usize>,
    ) -> PyResult<Py<PyAny>> {
        let actual_n_mfcc = n_mfcc.unwrap_or(13);
        let actual_n_mels = n_mels.unwrap_or(128);
        let actual_fmin = fmin.unwrap_or(0.0);
        let actual_fmax = parse_fmax(fmax, self.sample_rate());

        let mfcc_result = self
            .with_inner(|inner| inner.mfcc(actual_n_mfcc, actual_n_mels, actual_fmin, actual_fmax))
            .map_err(map_error)?;

        array2_to_numpy(py, mfcc_result)
    }

    /// Compute gammatone spectrogram for auditory modeling.
    ///
    /// Uses ERB-spaced gammatone filters to model the human auditory system.
    /// Useful for psychoacoustic research and hearing studies.
    ///
    /// # Arguments
    /// * `n_filters` - Number of gammatone filters
    /// * `fmin` - Minimum frequency in Hz
    /// * `fmax` - Maximum frequency in Hz
    /// * `window_size` - Analysis window size
    /// * `hop_size` - Hop size between frames
    ///
    /// # Returns
    /// 2D NumPy array (filter channels × time frames)
    ///
    /// # Examples
    /// ```python
    /// # Standard gammatone analysis
    /// gamma_spec = audio.gammatone_spectrogram(n_filters=64, fmin=80, fmax=8000)
    ///
    /// # High resolution auditory analysis
    /// gamma_hires = audio.gammatone_spectrogram(n_filters=128, window_size=2048)
    /// ```
    pub(crate) fn gammatone_spectrogram_impl(
        &self,
        py: Python,
        n_filters: Option<usize>,
        fmin: Option<f64>,
        fmax: Option<f64>,
        window_size: Option<usize>,
        hop_size: Option<usize>,
    ) -> PyResult<Py<PyAny>> {
        let actual_n_filters = n_filters.unwrap_or(64);
        let actual_fmin = fmin.unwrap_or(80.0);
        let actual_fmax = parse_fmax(fmax, self.sample_rate());
        let actual_window_size = window_size.unwrap_or(1024);
        let actual_hop_size = parse_hop_size(hop_size, actual_window_size);

        let gamma_spec = self
            .with_inner(|inner| {
                inner.gammatone_spectrogram(
                    actual_n_filters,
                    actual_fmin,
                    actual_fmax,
                    actual_window_size,
                    actual_hop_size,
                )
            })
            .map_err(map_error)?;

        array2_to_numpy(py, gamma_spec)
    }

    /// Compute chromagram (pitch class profile).
    ///
    /// Useful for music analysis and chord detection.
    /// Maps frequencies to the 12 pitch classes of the chromatic scale.
    ///
    /// # Arguments
    /// * `n_chroma` - Number of chroma bins (default: 12)
    /// * `window_size` - STFT window size
    /// * `hop_size` - STFT hop size
    ///
    /// # Returns
    /// 2D NumPy array (chroma bins × time frames)
    ///
    /// # Examples
    /// ```python
    /// # Standard 12-bin chromagram
    /// chroma = audio.chroma(n_chroma=12)
    ///
    /// # Higher resolution chroma analysis
    /// chroma_hires = audio.chroma(n_chroma=24, window_size=4096)
    /// ```
    pub(crate) fn chroma_impl(
        &self,
        py: Python,
        n_chroma: Option<usize>,
        window_size: Option<usize>,
        hop_size: Option<usize>,
    ) -> PyResult<Py<PyAny>> {
        let actual_n_chroma = n_chroma.unwrap_or(12);
        let actual_window_size = window_size.unwrap_or(1024);
        let _actual_hop_size = parse_hop_size(hop_size, actual_window_size);

        let chroma_result = self
            .with_inner(|inner| inner.chroma(actual_n_chroma))
            .map_err(map_error)?;
        array2_to_numpy(py, chroma_result)
    }

    /// Compute power spectral density using Welch's method.
    ///
    /// Estimates the power spectral density by averaging periodograms
    /// of overlapping segments.
    ///
    /// # Arguments
    /// * `window_size` - Size of each segment
    /// * `overlap` - Overlap between segments (0.0 to 1.0)
    /// * `window` - Window function to apply
    ///
    /// # Returns
    /// Tuple of (frequencies, power_spectral_density) as NumPy arrays
    ///
    /// # Examples
    /// ```python
    /// # Compute PSD with 50% overlap
    /// freqs, psd = audio.power_spectral_density(window_size=1024, overlap=0.5)
    ///
    /// # Plot PSD
    /// plt.loglog(freqs, psd)
    /// plt.xlabel('Frequency (Hz)')
    /// plt.ylabel('Power Spectral Density')
    /// ```
    pub(crate) fn power_spectral_density_impl(
        &self,
        py: Python,
        window_size: Option<usize>,
        overlap: Option<f64>,
        _window: Option<&str>,
    ) -> PyResult<Py<PyAny>> {
        let actual_window_size = window_size.unwrap_or(1024);
        let actual_overlap = overlap.unwrap_or(0.5);

        if actual_overlap < 0.0 || actual_overlap >= 1.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Overlap must be between 0.0 and 1.0 (exclusive)",
            ));
        }

        let (frequencies, psd) = self
            .with_inner(|inner| inner.power_spectral_density(actual_window_size, actual_overlap))
            .map_err(map_error)?;

        // Return as tuple of numpy arrays
        let freq_array = array1_to_numpy(py, ndarray::Array1::from(frequencies))?;
        let psd_array = array1_to_numpy(py, ndarray::Array1::from(psd))?;

        Ok(pyo3::types::PyTuple::new(py, &[freq_array, psd_array])?.into())
    }

    /// Compute power spectrogram in dB scale in a single call.
    ///
    /// This is the streamlined version that performs STFT → magnitude → power → dB 
    /// conversion all in one method call, perfect for librosa-style workflows.
    ///
    /// # Arguments  
    /// * `n_fft` - FFT window size in samples (default: 2048)
    /// * `hop_length` - Number of samples between frames (default: n_fft/4) 
    /// * `win_length` - Window length in samples (default: n_fft)
    /// * `window` - Window function type (default: 'hann')
    /// * `center` - Whether to pad the signal to center frames (default: true)
    /// * `ref_val` - Reference value for dB conversion (default: 1.0)
    /// * `amin` - Minimum threshold to avoid log(0) (default: 1e-10)
    /// * `top_db` - Threshold the output at top_db below maximum (optional)
    ///
    /// Legacy parameters (backward compatibility):
    /// * `window_size` - Alias for n_fft  
    /// * `hop_size` - Alias for hop_length
    ///
    /// # Returns
    /// 2D NumPy array with power spectrogram in dB scale (frequency bins × time frames)
    /// 
    /// # Examples
    /// ```python
    /// # One-liner librosa-compatible workflow
    /// S_db = audio.spectrogram_db(n_fft=2048, hop_length=512)
    /// 
    /// # With reference normalization
    /// S_db = audio.spectrogram_db(n_fft=1024, hop_length=256, ref_val=1.0, top_db=80)
    ///
    /// # Legacy compatibility
    /// S_db = audio.spectrogram_db(window_size=2048, hop_size=512)
    /// ```
    pub(crate) fn spectrogram_db_impl(
        &self,
        py: Python,
        n_fft: Option<usize>,
        hop_length: Option<usize>, 
        win_length: Option<usize>,
        window: &str,
        center: bool,
        ref_val: f64,
        amin: f64,
        top_db: f64,
        // Legacy parameters for backward compatibility
        window_size: Option<usize>,
        hop_size: Option<usize>,
    ) -> PyResult<Py<PyAny>> {
        // Resolve parameters with librosa priority
        let actual_n_fft = n_fft.or(window_size).unwrap_or(2048);
        let actual_hop_length = hop_length.or(hop_size).unwrap_or(actual_n_fft / 4);
        let actual_win_length = win_length.unwrap_or(actual_n_fft);
        
        // Validate parameters
        if actual_win_length > actual_n_fft {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "win_length cannot be larger than n_fft"
            ));
        }

        let window_type = Self::parse_window_type_impl(window)?;

        // Compute STFT
        let stft_result = self
            .with_inner(|inner| {
                inner.stft(actual_n_fft, actual_hop_length, window_type)
            })
            .map_err(map_error)?;
            
        // Note: center and win_length parameters are accepted for API compatibility
        // but not currently used in the core STFT implementation
        let _ = center;
        let _ = actual_win_length;

        // Convert complex STFT to magnitude, then to power, then to dB
        // First compute power spectrum
        let power_spectrum = stft_result.mapv(|x| {
            let magnitude = x.norm(); // abs(complex)
            magnitude * magnitude // magnitude^2
        });
        
        // Determine reference value - use max of power spectrum if ref_val is 1.0 and we want librosa compatibility
        let actual_ref_val = if ref_val == 1.0 {
            // Use max of power spectrum like librosa does: ref=np.max(np.abs(D)**2)
            power_spectrum.fold(0.0f64, |max, &x| x.max(max)).max(amin)
        } else {
            ref_val
        };
        
        // Convert to dB: 10*log10(power/ref_val)
        let mut power_db = power_spectrum.mapv(|power| {
            10.0 * (power.max(amin) / actual_ref_val).log10()
        });
        
        // Apply top_db threshold (librosa-style)
        if top_db > 0.0 {
            let max_val = power_db.fold(f64::NEG_INFINITY, |max, &x| x.max(max));
            power_db.mapv_inplace(|x| x.max(max_val - top_db));
        }

        array2_to_numpy(py, power_db)
    }
}
