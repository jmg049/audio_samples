//! FFT backend selection for optimal performance across different file sizes
//!
//! This module provides size-aware FFT backend selection to maximize performance:
//! - Small files (≤5 seconds): Use realfft
//! - Large files (>5 seconds): Use Intel MKL FFTW interface (must be enabled via "mkl" feature)

use std::{collections::HashMap, sync::Arc};

use crate::{
    AudioSampleError, AudioSampleResult, LayoutError, ProcessingError, RealFloat, to_precision,
};
use num_complex::Complex;
#[cfg(feature = "fft")]
use realfft::{RealFftPlanner, RealToComplex};

/// FFT backend selection strategies
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FftBackend {
    /// Pure Rust realfft implementation (fast for small files)
    RealFFT,
    /// Intel MKL via FFTW interface (optimized for large files)
    #[cfg(feature = "mkl")]
    IntelMKL,
}

/// Automatic backend selection based on audio duration and processing requirements
pub fn select_fft_backend<F: RealFloat>(duration_seconds: F, total_samples: usize) -> FftBackend {
    select_fft_backend_with_sample_rate(duration_seconds, total_samples, None)
}

/// Advanced backend selection with sample rate awareness
pub fn select_fft_backend_with_sample_rate<F: RealFloat>(
    duration_seconds: F,
    total_samples: usize,
    sample_rate: Option<u32>,
) -> FftBackend {
    // Adaptive thresholds based on sample rate and processing characteristics
    let (duration_threshold, samples_threshold) =
        calculate_adaptive_thresholds(sample_rate, total_samples, duration_seconds);

    #[cfg(feature = "mkl")]
    {
        // Use MKL for large files when available
        if duration_seconds > duration_threshold || total_samples > samples_threshold {
            FftBackend::IntelMKL
        } else {
            FftBackend::RealFFT
        }
    }

    #[cfg(not(feature = "mkl"))]
    {
        // Always use RealFFT when MKL is not available
        let _ = (duration_seconds, total_samples, sample_rate); // Suppress unused warnings
        FftBackend::RealFFT
    }
}

/// Calculate optimal thresholds based on audio characteristics
///
/// Determined via trial and error when testing
fn calculate_adaptive_thresholds<F: RealFloat>(
    sample_rate: Option<u32>,
    total_samples: usize,
    duration_seconds: F,
) -> (F, usize) {
    // Base thresholds optimized for different sample rates
    let (base_duration, base_samples) = match sample_rate {
        Some(sr) => match sr {
            // Low sample rates (8kHz-16kHz): Earlier MKL switch
            sr if sr <= 16000 => (2.0, 32_000), // ~2s at 16kHz
            // Medium sample rates (22kHz-32kHz)
            sr if sr <= 32000 => (2.5, 80_000), // ~2.5s at 32kHz
            // CD quality (44.1kHz)
            sr if sr <= 44100 => (3.0, 132_300), // ~3s at 44.1kHz
            // High sample rates (48kHz+): Later MKL switch due to larger overhead
            sr if sr <= 48000 => (3.5, 168_000), // ~3.5s at 48kHz
            sr if sr <= 96000 => (4.0, 384_000), // ~4s at 96kHz
            _ => (5.0, 500_000),                 // Very high sample rates
        },
        // No sample rate info: use conservative defaults
        None => {
            // When no sample rate is provided, be more conservative about switching to MKL
            // Estimate based on duration vs samples ratio to infer sample rate
            let estimated_sr = if duration_seconds > F::zero() {
                (to_precision::<F, _>(total_samples) / duration_seconds)
                    .to_u32()
                    .unwrap_or(44100)
            } else {
                44100 // Default assumption
            };

            match estimated_sr {
                sr if sr <= 16000 => (5.0, 80_000),  // Low sample rate
                sr if sr <= 32000 => (5.0, 160_000), // Medium sample rate
                sr if sr <= 44100 => (5.0, 220_500), // CD quality (~5s at 44.1kHz)
                sr if sr <= 48000 => (5.0, 240_000), // High quality (~5s at 48kHz)
                sr if sr <= 96000 => (5.0, 480_000), // Very high quality (~5s at 96kHz)
                _ => (5.0, 500_000),                 // Ultra high quality
            }
        }
    };
    let base_duration = to_precision::<F, _>(base_duration);
    // Additional adjustments based on total samples for very large files
    let adjusted_duration = if total_samples > 5_000_000 {
        // >5M samples (~113s at 44.1kHz)
        // For extremely large files, switch to MKL earlier
        base_duration * to_precision::<F, _>(0.7)
    } else if total_samples > 1_000_000 {
        // >1M samples (~23s at 44.1kHz)
        // For large files, slightly earlier switch
        base_duration * to_precision::<F, _>(0.8)
    } else {
        base_duration
    };

    (adjusted_duration, base_samples)
}

/// Trait for FFT backend implementations
pub trait FftBackendImpl<F: RealFloat> {
    /// Performs real-to-complex FFT
    fn compute_real_fft(&mut self, input: &[F], output: &mut [Complex<F>])
    -> AudioSampleResult<()>;
}

/// Returns the expected output size for given input size
pub(crate) const fn output_size(input_size: usize) -> usize {
    input_size / 2 + 1
}

/// RealFFT backend implementation
pub struct RealFftBackend<F: RealFloat> {
    planner: RealFftPlanner<F>,
    cached_plans: HashMap<usize, Arc<dyn RealToComplex<F>>>,
}

impl<F: RealFloat> RealFftBackend<F> {
    /// Creates a new real FFT backend with an empty plan cache.
    pub fn new() -> Self {
        Self {
            planner: RealFftPlanner::new(),
            cached_plans: HashMap::new(),
        }
    }

    fn get_or_create_plan(&mut self, size: usize) -> std::sync::Arc<dyn RealToComplex<F>> {
        if let Some(plan) = self.cached_plans.get(&size) {
            plan.clone()
        } else {
            let plan = self.planner.plan_fft_forward(size);
            self.cached_plans.insert(size, plan.clone());
            plan
        }
    }
}

impl<F: RealFloat> Default for RealFftBackend<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: RealFloat> FftBackendImpl<F> for RealFftBackend<F> {
    fn compute_real_fft(
        &mut self,
        input: &[F],
        output: &mut [Complex<F>],
    ) -> AudioSampleResult<()> {
        let plan = self.get_or_create_plan(input.len());

        // RealFFT requires mutable input, so we need to copy
        let mut input_copy: Vec<F> = input.to_vec();

        plan.process(&mut input_copy, output).map_err(|e| {
            AudioSampleError::Processing(ProcessingError::algorithm_failure(
                "realfft",
                format!("RealFFT error: {:?}", e),
            ))
        })
    }
}

/// Intel MKL backend implementation via FFTW interface
#[cfg(feature = "mkl")]
pub struct MklFftBackend {
    cached_plans: HashMap<usize, MklPlan>,
}

#[cfg(feature = "mkl")]
struct MklPlan {
    plan: fftw_sys::fftw_plan,
    #[allow(dead_code)]
    input_size: usize,
    output_size: usize,
}

#[cfg(feature = "mkl")]
impl MklFftBackend {
    /// Creates a new MKL FFT backend with an empty plan cache.
    pub fn new() -> AudioSampleResult<Self> {
        Ok(Self {
            cached_plans: HashMap::new(),
        })
    }

    fn get_or_create_plan(&mut self, size: usize) -> AudioSampleResult<&MklPlan> {
        if let std::collections::hash_map::Entry::Vacant(e) = self.cached_plans.entry(size) {
            let output_size = output_size(size);

            // Create FFTW plan for real-to-complex FFT
            let plan = unsafe {
                let input = fftw_sys::fftw_malloc(std::mem::size_of::<f64>() * size) as *mut f64;
                let output = fftw_sys::fftw_malloc(
                    std::mem::size_of::<fftw_sys::fftw_complex>() * output_size,
                ) as *mut fftw_sys::fftw_complex;

                if input.is_null() || output.is_null() {
                    if !input.is_null() {
                        fftw_sys::fftw_free(input as *mut std::ffi::c_void);
                    }
                    if !output.is_null() {
                        fftw_sys::fftw_free(output as *mut std::ffi::c_void);
                    }
                    return Err(AudioSampleError::Processing(
                        ProcessingError::algorithm_failure(
                            "fftw",
                            "Failed to allocate FFTW memory",
                        ),
                    ));
                }

                let plan = fftw_sys::fftw_plan_dft_r2c_1d(
                    size as i32,
                    input,
                    output,
                    fftw_sys::FFTW_ESTIMATE,
                );

                fftw_sys::fftw_free(input as *mut std::ffi::c_void);
                fftw_sys::fftw_free(output as *mut std::ffi::c_void);

                if plan.is_null() {
                    return Err(AudioSampleError::Processing(
                        ProcessingError::algorithm_failure("fftw", "Failed to create FFTW plan"),
                    ));
                }

                plan
            };

            let mkl_plan = MklPlan {
                plan,
                input_size: size,
                output_size,
            };

            e.insert(mkl_plan);
        }

        Ok(&self.cached_plans[&size])
    }
}

#[cfg(feature = "mkl")]
impl Default for MklFftBackend {
    fn default() -> Self {
        Self::new().expect("Failed to create MKL FFT backend")
    }
}

#[cfg(feature = "mkl")]
impl Drop for MklFftBackend {
    fn drop(&mut self) {
        unsafe {
            for plan_info in self.cached_plans.values() {
                fftw_sys::fftw_destroy_plan(plan_info.plan);
            }
        }
    }
}

#[cfg(feature = "mkl")]
impl<F: RealFloat> FftBackendImpl<F> for MklFftBackend {
    fn compute_real_fft(
        &mut self,
        input: &[F],
        output: &mut [Complex<F>],
    ) -> AudioSampleResult<()> {
        let plan_info = self.get_or_create_plan(input.len())?;

        if output.len() != plan_info.output_size {
            return Err(AudioSampleError::Layout(LayoutError::dimension_mismatch(
                format!("FFT output size {}", plan_info.output_size),
                format!("provided output size {}", output.len()),
                "FFT computation",
            )));
        }

        unsafe {
            // Allocate aligned input and output arrays
            let input_ptr = fftw_sys::fftw_malloc(std::mem::size_of_val(input)) as *mut F;
            let output_ptr =
                fftw_sys::fftw_malloc(std::mem::size_of::<fftw_sys::fftw_complex>() * output.len())
                    as *mut fftw_sys::fftw_complex;

            if input_ptr.is_null() || output_ptr.is_null() {
                if !input_ptr.is_null() {
                    fftw_sys::fftw_free(input_ptr as *mut std::ffi::c_void);
                }
                if !output_ptr.is_null() {
                    fftw_sys::fftw_free(output_ptr as *mut std::ffi::c_void);
                }
                return Err(AudioSampleError::Processing(
                    ProcessingError::algorithm_failure(
                        "fftw",
                        "Failed to allocate aligned FFTW memory",
                    ),
                ));
            }

            // Copy input data
            std::ptr::copy_nonoverlapping(input.as_ptr(), input_ptr, input.len());

            let input_ptr =
                fftw_sys::fftw_malloc(std::mem::size_of::<f64>() * input.len()) as *mut f64;
            // Execute FFT with the cached plan but new data
            fftw_sys::fftw_execute_dft_r2c(plan_info.plan, input_ptr, output_ptr);

            // Convert FFTW complex format to num_complex format
            // FFTW uses [f64; 2] where [0] is real, [1] is imaginary
            for (i, out_val) in output.iter_mut().enumerate() {
                use crate::to_precision;

                let complex_val = std::slice::from_raw_parts(output_ptr.add(i) as *const f64, 2);
                *out_val = Complex::new(
                    to_precision::<F, _>(complex_val[0]),
                    to_precision::<F, _>(complex_val[1]),
                );
            }

            // Free allocated memory
            fftw_sys::fftw_free(input_ptr as *mut std::ffi::c_void);
            fftw_sys::fftw_free(output_ptr as *mut std::ffi::c_void);
        }

        Ok(())
    }
}

/// Unified FFT backend that automatically selects the optimal implementation
pub enum UnifiedFftBackend<F: RealFloat> {
    /// RealFFT backend using the realfft crate
    RealFFT(Box<RealFftBackend<F>>),
    /// Intel MKL backend for optimized performance
    #[cfg(feature = "mkl")]
    IntelMKL(MklFftBackend),
}

impl<F: RealFloat> UnifiedFftBackend<F> {
    /// Creates a new unified FFT backend using the specified implementation.
    ///
    /// # Arguments
    /// * `backend` - The FFT backend type to use
    ///
    /// # Returns
    /// A new `UnifiedFftBackend` instance
    pub fn new(backend: FftBackend) -> AudioSampleResult<Self> {
        match backend {
            FftBackend::RealFFT => Ok(Self::RealFFT(Box::default())),
            #[cfg(feature = "mkl")]
            FftBackend::IntelMKL => Ok(Self::IntelMKL(MklFftBackend::new()?)),
        }
    }

    /// Automatically selects the optimal FFT backend based on audio characteristics.
    ///
    /// # Arguments
    /// * `duration_seconds` - Duration of the audio in seconds
    /// * `total_samples` - Total number of samples in the audio
    ///
    /// # Returns
    /// A new `UnifiedFftBackend` instance with the optimal backend selected
    pub fn auto_select(duration_seconds: F, total_samples: usize) -> AudioSampleResult<Self> {
        let backend = select_fft_backend(duration_seconds, total_samples);
        Self::new(backend)
    }

    /// Automatically selects the optimal FFT backend considering sample rate.
    ///
    /// # Arguments
    /// * `duration_seconds` - Duration of the audio in seconds
    /// * `total_samples` - Total number of samples in the audio
    /// * `sample_rate` - Optional sample rate for additional optimization hints
    ///
    /// # Returns
    /// A new `UnifiedFftBackend` instance with the optimal backend selected
    pub fn auto_select_with_sample_rate(
        duration_seconds: f64,
        total_samples: usize,
        sample_rate: Option<u32>,
    ) -> AudioSampleResult<Self> {
        let backend =
            select_fft_backend_with_sample_rate(duration_seconds, total_samples, sample_rate);
        Self::new(backend)
    }
}

impl<F: RealFloat> FftBackendImpl<F> for UnifiedFftBackend<F> {
    fn compute_real_fft(
        &mut self,
        input: &[F],
        output: &mut [Complex<F>],
    ) -> AudioSampleResult<()> {
        match self {
            Self::RealFFT(backend) => backend.compute_real_fft(input, output),
            #[cfg(feature = "mkl")]
            Self::IntelMKL(backend) => backend.compute_real_fft(input, output),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_selection() {
        // Small files should use RealFFT
        assert_eq!(select_fft_backend(1.0, 44100), FftBackend::RealFFT);
        assert_eq!(select_fft_backend(4.9, 200_000), FftBackend::RealFFT);

        // Large files should use MKL when available
        #[cfg(feature = "mkl")]
        {
            assert_eq!(select_fft_backend(10.0, 441_000), FftBackend::IntelMKL);
            assert_eq!(select_fft_backend(5.1, 250_000), FftBackend::IntelMKL);
        }

        #[cfg(not(feature = "mkl"))]
        {
            // Without MKL, should always use RealFFT
            assert_eq!(select_fft_backend(10.0, 441_000), FftBackend::RealFFT);
            assert_eq!(select_fft_backend(5.1, 250_000), FftBackend::RealFFT);
        }
    }

    #[test]
    fn test_output_size_calculation() {
        assert_eq!(output_size(1024), 513);
        assert_eq!(output_size(2048), 1025);
        assert_eq!(output_size(512), 257);
    }
}
